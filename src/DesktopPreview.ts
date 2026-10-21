import type { Mat4, Vec3 } from './math'
import type { PointerInput } from './Scene'
import type Scene from './Scene'
import type SceneRenderer from './SceneRenderer'
import { invert, multiply, normalize, perspective, poseFromRay, transformPoint } from './math'

// XR 非対応環境向けのプレビュー。
// 左ドラッグで視点回転、右ドラッグ (Shift + 左ドラッグ) でフィールドを掴んで移動、
// ホイールで拡縮、WASD / QE で空間内を移動
export default class DesktopPreview {
  private canvas: HTMLCanvasElement
  private context: GPUCanvasContext
  private renderer: SceneRenderer
  private scene: Scene
  private format: GPUTextureFormat

  private eye: Vec3 = [0, 1.6, 0]
  private yaw = 0
  private pitch = -0.05
  private mouse: [number, number] = [0, 0]
  private mouseInside = false
  private mouseDown = false
  private grabbing = false
  private keys = new Set<string>()
  private pendingScale = 0
  private readonly pointerKey = {}

  private frameId: number | null = null
  private lastTime: number | null = null

  constructor(canvas: HTMLCanvasElement, device: GPUDevice, renderer: SceneRenderer, scene: Scene) {
    this.canvas = canvas
    this.renderer = renderer
    this.scene = scene
    this.context = canvas.getContext('webgpu') as GPUCanvasContext
    this.format = navigator.gpu.getPreferredCanvasFormat()
    this.context.configure({ device, format: this.format, alphaMode: 'opaque' })

    new ResizeObserver(() => {
      const dpr = Math.min(window.devicePixelRatio, 2)
      canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr))
      canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr))
    }).observe(canvas)

    canvas.addEventListener('pointermove', (e) => {
      this.mouseInside = true
      const [lastX, lastY] = this.mouse
      this.mouse = [e.offsetX, e.offsetY]
      if (this.mouseDown && !this.grabbing) {
        this.yaw -= (e.offsetX - lastX) * 0.004
        this.pitch = Math.min(Math.max(this.pitch - (e.offsetY - lastY) * 0.004, -1.4), 1.4)
      }
    })
    canvas.addEventListener('pointerleave', () => {
      this.mouseInside = false
    })
    canvas.addEventListener('pointerdown', (e) => {
      canvas.setPointerCapture(e.pointerId)
      this.mouseInside = true
      this.mouse = [e.offsetX, e.offsetY]
      this.mouseDown = true
      this.grabbing = e.button === 2 || e.shiftKey
    })
    const release = () => {
      this.mouseDown = false
      this.grabbing = false
    }
    canvas.addEventListener('contextmenu', e => e.preventDefault())
    window.addEventListener('keydown', e => this.keys.add(e.code))
    window.addEventListener('keyup', e => this.keys.delete(e.code))
    window.addEventListener('blur', () => this.keys.clear())
    canvas.addEventListener('pointerup', release)
    canvas.addEventListener('pointercancel', release)
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault()
      this.pendingScale += e.deltaY
    }, { passive: false })
  }

  private cameraMatrix(): Mat4 {
    const cy = Math.cos(this.yaw)
    const sy = Math.sin(this.yaw)
    const cp = Math.cos(this.pitch)
    const sp = Math.sin(this.pitch)
    const forward: Vec3 = [-sy * cp, sp, -cy * cp]
    return poseFromRay(this.eye, forward)
  }

  start() {
    if (this.frameId !== null)
      return
    this.lastTime = null
    const frame = (time: number) => {
      this.frameId = requestAnimationFrame(frame)
      this.render(time)
    }
    this.frameId = requestAnimationFrame(frame)
  }

  stop() {
    if (this.frameId !== null)
      cancelAnimationFrame(this.frameId)
    this.frameId = null
  }

  private render(time: number) {
    const dt = this.lastTime === null ? 0 : Math.min((time - this.lastTime) * 0.001, 0.1)
    this.lastTime = time

    const { canvas } = this
    this.move(dt)
    const camera = this.cameraMatrix()
    const proj = perspective(Math.PI / 3, canvas.width / canvas.height, 0.05, 100)
    const viewProj = multiply(proj, invert(camera))

    // マウス位置をワールド空間のレイに変換
    const pointers: PointerInput[] = []
    if (this.mouseInside || this.mouseDown) {
      const ndcX = (this.mouse[0] / canvas.clientWidth) * 2 - 1
      const ndcY = 1 - (this.mouse[1] / canvas.clientHeight) * 2
      const invViewProj = invert(viewProj)
      const unproject = (z: number): Vec3 => {
        const p = transformPoint(invViewProj, [ndcX, ndcY, z])
        const w = invViewProj[3] * ndcX + invViewProj[7] * ndcY + invViewProj[11] * z + invViewProj[15]
        return [p[0] / w, p[1] / w, p[2] / w]
      }
      const near = unproject(0)
      const far = unproject(1)
      const dir = normalize([far[0] - near[0], far[1] - near[1], far[2] - near[2]])
      pointers.push({
        key: this.pointerKey,
        rayPose: poseFromRay(this.eye, dir),
        pressed: this.grabbing,
        scaleAxis: 0,
        cursorDistance: 1.2,
        drawCursor: true,
      })
    }
    this.scene.update(dt, pointers)
    if (this.pendingScale !== 0) {
      this.scene.setScale(this.scene.fieldScale * Math.exp(-this.pendingScale * 0.001))
      this.pendingScale = 0
    }

    this.renderer.render(
      [{
        texture: this.context.getCurrentTexture(),
        views: [{ viewProj, cameraPosition: this.eye }],
      }],
      this.scene.frameData({ r: 0.02, g: 0.02, b: 0.04, a: 1 }),
    )
  }

  private move(dt: number) {
    const speed = 1.5 * dt
    const k = (code: string) => (this.keys.has(code) ? 1 : 0)
    const forward = k('KeyW') - k('KeyS')
    const right = k('KeyD') - k('KeyA')
    const up = k('KeyE') - k('KeyQ')
    const sy = Math.sin(this.yaw)
    const cy = Math.cos(this.yaw)
    this.eye = [
      this.eye[0] + (-sy * forward + cy * right) * speed,
      this.eye[1] + up * speed,
      this.eye[2] + (-cy * forward - sy * right) * speed,
    ]
  }

  resetField() {
    this.scene.placeInFront(this.cameraMatrix())
  }
}
