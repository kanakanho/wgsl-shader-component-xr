import type { Mat4, Vec3 } from './math'
import type { FrameData } from './SceneRenderer'
import {
  getPosition,
  invert,
  multiply,
  normalize,
  poseFromRay,
  scaling,
  transformDir,
  transformPoint,
  translation,
} from './math'
import { MAX_INSTANCES } from './SceneRenderer'

export interface SceneParams {
  bpm: number
  r: number
  g: number
  b: number
}

// 6DoF の入力 1 つ分 (XR コントローラー / ハンド / マウス)
export interface PointerInput {
  key: object
  rayPose: Mat4 // 原点がレイの始点、-Z がレイ方向
  gripPose?: Mat4 | null // コントローラー本体の姿勢
  pressed: boolean
  scaleAxis: number // -1..1 フィールド拡縮
  cursorDistance: number // レイ上のどこを「先端」としてフィールドに作用させるか (m)
  drawCursor: boolean
}

interface Grab {
  key: object
  offset: Mat4 // レイ姿勢から見たフィールド姿勢
}

interface TwoHandScale {
  key: object
  startDistance: number
  startScale: number
}

const MIN_SCALE = 0.1
const MAX_SCALE = 10

export default class Scene {
  params: SceneParams
  arMode = false
  floorY: number | null = 0

  // SDF フィールドの姿勢と 1 単位あたりのメートル数
  fieldPose: Mat4 = translation(0, 1.4, -1.5)
  fieldScale = 1

  private grab: Grab | null = null
  private twoHand: TwoHandScale | null = null
  private wasPressed = new WeakMap<object, boolean>()
  private pointers: PointerInput[] = []
  private startTime = performance.now()

  constructor(params: SceneParams) {
    this.params = params
  }

  // 視点の正面 distance の位置にフィールドの中心を置く
  placeInFront(viewerPose: Mat4, distance: number = 1.5) {
    const pos = getPosition(viewerPose)
    const forward = transformDir(viewerPose, [0, 0, -1])
    const flat = normalize([forward[0], 0, forward[2]])
    const center: Vec3 = [
      pos[0] + flat[0] * distance,
      pos[1] - 0.2,
      pos[2] + flat[2] * distance,
    ]
    this.fieldPose = poseFromRay(center, flat)
    this.grab = null
    this.twoHand = null
  }

  isGrabbing(key: object): boolean {
    return this.grab?.key === key
  }

  private cursor(p: PointerInput): Vec3 {
    return transformPoint(p.rayPose, [0, 0, -p.cursorDistance])
  }

  update(dt: number, pointers: PointerInput[]) {
    this.pointers = pointers
    const find = (key: object) => pointers.find(p => p.key === key && p.pressed)

    // 離した入力の掴みを解放
    if (this.grab && !find(this.grab.key))
      this.grab = null
    if (this.twoHand && (!this.grab || !find(this.twoHand.key)))
      this.twoHand = null

    for (const p of pointers) {
      const justPressed = p.pressed && !this.wasPressed.get(p.key)
      this.wasPressed.set(p.key, p.pressed)
      if (!justPressed)
        continue

      if (!this.grab) {
        // 1 つ目: フィールドを掴む (6DoF で移動・回転)
        this.grab = { key: p.key, offset: multiply(invert(p.rayPose), this.fieldPose) }
      }
      else if (!this.twoHand && p.key !== this.grab.key) {
        // 2 つ目: 両手の距離で拡縮
        const first = find(this.grab.key)!
        this.twoHand = {
          key: p.key,
          startDistance: Math.max(distance(this.cursor(first), this.cursor(p)), 1e-3),
          startScale: this.fieldScale,
        }
      }
    }

    if (this.grab) {
      const holder = find(this.grab.key)!
      this.fieldPose = multiply(holder.rayPose, this.grab.offset)

      if (this.twoHand) {
        const other = find(this.twoHand.key)!
        const d = distance(this.cursor(holder), this.cursor(other))
        this.setScale(this.twoHand.startScale * d / this.twoHand.startDistance)
      }
      else if (Math.abs(holder.scaleAxis) > 0.1) {
        this.setScale(this.fieldScale * (1 - holder.scaleAxis * dt))
      }
    }
  }

  setScale(scale: number) {
    this.fieldScale = Math.min(Math.max(scale, MIN_SCALE), MAX_SCALE)
  }

  frameData(clearColor: GPUColor): FrameData {
    const { bpm, r, g, b } = this.params
    const elapsed = (performance.now() - this.startTime) * 0.001
    const model = multiply(this.fieldPose, scaling(this.fieldScale, this.fieldScale, this.fieldScale))

    const sceneUniform = new Float32Array(32)
    sceneUniform.set(invert(model), 0)
    sceneUniform[16] = elapsed * (bpm / 60)
    sceneUniform[17] = this.fieldScale
    sceneUniform[18] = this.floorY ?? 0
    sceneUniform[19] = this.arMode ? 1 : 0
    sceneUniform.set([r / 255, g / 255, b / 255, this.floorY !== null && !this.arMode ? 1 : 0], 20)

    const instanceData = new Float32Array(MAX_INSTANCES * 20)
    let instanceCount = 0
    const pushInstance = (m: Mat4, color: [number, number, number, number]) => {
      if (instanceCount >= MAX_INSTANCES)
        return
      instanceData.set(m, instanceCount * 20)
      instanceData.set(color, instanceCount * 20 + 16)
      instanceCount++
    }

    let slot = 0
    for (const p of this.pointers) {
      const cursor = this.cursor(p)
      if (slot < 2) {
        sceneUniform.set([...cursor, p.pressed ? 2 : 1], 24 + slot * 4)
        slot++
      }
      if (p.gripPose)
        pushInstance(multiply(p.gripPose, scaling(0.03, 0.03, 0.1)), [0.8, 0.8, 0.85, 1])
      if (p.drawCursor) {
        const size = p.pressed ? 0.02 : 0.012
        const color: [number, number, number, number] = this.isGrabbing(p.key)
          ? [1, 0.6, 0.2, 1]
          : [0.6, 0.85, 1, 1]
        pushInstance(multiply(translation(...cursor), scaling(size, size, size)), color)
      }
    }

    return { sceneUniform, instanceData, instanceCount, clearColor }
  }
}

function distance(a: Vec3, b: Vec3) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
}
