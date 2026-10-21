import type Scene from './Scene'
import type { PointerInput } from './Scene'
import type SceneRenderer from './SceneRenderer'
import type { RenderTarget } from './SceneRenderer'
import { getPosition, multiply } from './math'

export type XRMode = 'immersive-vr' | 'immersive-ar'

export function isXRGPUBindingSupported(): boolean {
  return typeof window.XRGPUBinding === 'function'
}

export async function isXRModeSupported(mode: XRMode): Promise<boolean> {
  if (!navigator.xr)
    return false
  try {
    return await navigator.xr.isSessionSupported(mode)
  }
  catch {
    return false
  }
}

// WebXR セッションを開始し、WebGPU (XRGPUBinding) で各ビューへ描画する
export async function startXRSession(
  mode: XRMode,
  device: GPUDevice,
  renderer: SceneRenderer,
  scene: Scene,
  onEnd: () => void,
): Promise<XRSession> {
  if (!navigator.xr)
    throw new Error('WebXR is not available.')
  if (!window.XRGPUBinding)
    throw new Error('XRGPUBinding (WebXR/WebGPU binding) is not available.')

  const session = await navigator.xr.requestSession(mode, {
    requiredFeatures: ['webgpu'],
    optionalFeatures: ['local-floor', 'hand-tracking'],
  })

  let refSpace: XRReferenceSpace
  let hasFloor = true
  try {
    refSpace = await session.requestReferenceSpace('local-floor')
  }
  catch {
    refSpace = await session.requestReferenceSpace('local')
    hasFloor = false
  }

  const binding = new window.XRGPUBinding(session, device)
  const layer = binding.createProjectionLayer({
    colorFormat: binding.getPreferredColorFormat(),
  })
  session.updateRenderState({ layers: [layer], depthNear: 0.05, depthFar: 100 })

  const isAR = mode === 'immersive-ar'
  scene.arMode = isAR
  scene.floorY = hasFloor ? 0 : null

  // select (トリガー / ピンチ) と squeeze (グリップ) の押下状態
  const pressed = new Map<XRInputSource, Set<string>>()
  const press = (type: string, down: boolean) => (event: XRInputSourceEvent) => {
    const set = pressed.get(event.inputSource) ?? new Set<string>()
    if (down)
      set.add(type)
    else
      set.delete(type)
    pressed.set(event.inputSource, set)
  }
  session.addEventListener('selectstart', press('select', true))
  session.addEventListener('selectend', press('select', false))
  session.addEventListener('squeezestart', press('squeeze', true))
  session.addEventListener('squeezeend', press('squeeze', false))
  session.addEventListener('inputsourceschange', (event) => {
    for (const source of event.removed)
      pressed.delete(source)
  })

  let needsPlacement = true
  let lastTime: number | null = null

  const onFrame: XRFrameRequestCallback = (time, frame) => {
    session.requestAnimationFrame(onFrame)
    const dt = lastTime === null ? 0 : Math.min((time - lastTime) * 0.001, 0.1)
    lastTime = time

    const viewerPose = frame.getViewerPose(refSpace)
    if (!viewerPose)
      return

    if (needsPlacement) {
      scene.placeInFront(viewerPose.transform.matrix)
      needsPlacement = false
    }

    // 6DoF 入力の取得
    const pointers: PointerInput[] = []
    for (const source of session.inputSources) {
      const rayPose = frame.getPose(source.targetRaySpace, refSpace)
      if (!rayPose)
        continue
      const gripPose = source.gripSpace ? frame.getPose(source.gripSpace, refSpace) : null
      const axes = source.gamepad?.axes ?? []
      pointers.push({
        key: source,
        rayPose: rayPose.transform.matrix,
        gripPose: source.hand ? null : gripPose?.transform.matrix ?? null,
        pressed: (pressed.get(source)?.size ?? 0) > 0,
        scaleAxis: axes[3] ?? 0,
        // コントローラー / 手の少し先をフィールドに作用させる点とする
        cursorDistance: source.targetRayMode === 'screen' ? 1.0 : 0.08,
        drawCursor: source.targetRayMode !== 'screen',
      })
    }
    scene.update(dt, pointers)

    // 同じテクスチャ・レイヤーを共有するビューは 1 つのレンダーパスにまとめる
    const targets: RenderTarget[] = []
    for (const view of viewerPose.views) {
      const subImage = binding.getViewSubImage(layer, view)
      const viewDescriptor = subImage.getViewDescriptor()
      const layerIndex = viewDescriptor.baseArrayLayer ?? 0
      let target = targets.find(t =>
        t.texture === subImage.colorTexture
        && (t.viewDescriptor?.baseArrayLayer ?? 0) === layerIndex)
      if (!target) {
        target = { texture: subImage.colorTexture, viewDescriptor, views: [] }
        targets.push(target)
      }
      const { x, y, width, height } = subImage.viewport
      target.views.push({
        viewProj: multiply(view.projectionMatrix, view.transform.inverse.matrix),
        cameraPosition: getPosition(view.transform.matrix),
        viewport: { x, y, width, height },
      })
    }

    const clearColor: GPUColor = isAR
      ? { r: 0, g: 0, b: 0, a: 0 }
      : { r: 0.02, g: 0.02, b: 0.04, a: 1 }
    renderer.render(targets, scene.frameData(clearColor))
  }

  session.addEventListener('end', () => {
    scene.arMode = false
    scene.floorY = 0
    onEnd()
  })
  session.requestAnimationFrame(onFrame)
  return session
}
