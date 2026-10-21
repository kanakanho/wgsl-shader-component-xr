import type { XRMode } from './XRSessionRunner'
import DesktopPreview from './DesktopPreview'
import Scene from './Scene'
import SceneRenderer from './SceneRenderer'
import { isXRGPUBindingSupported, isXRModeSupported, startXRSession } from './XRSessionRunner'
import './style.css'

const app = document.querySelector<HTMLDivElement>('#app')!
app.innerHTML = `
  <canvas id="preview"></canvas>
  <div class="panel">
    <div class="buttons">
      <button id="enter-vr" disabled>Enter VR</button>
      <button id="enter-ar" disabled>Enter AR</button>
    </div>
    <label>BPM <input id="bpm" type="range" min="30" max="240" value="120"><span id="bpm-value">120</span></label>
    <label>Color <input id="color" type="color" value="#ffffff"></label>
    <p id="status"></p>
    <p class="hint">
      XR: トリガー / ピンチで空間 (SDF フィールド) を掴んで移動・回転、両手で掴んで拡縮、掴んだままスティック上下でも拡縮<br>
      プレビュー: 左ドラッグで視点回転、右ドラッグ (Shift+ドラッグ) でフィールドを掴む、ホイールで拡縮、WASD / QE で移動
    </p>
  </div>
`

const status = document.querySelector<HTMLParagraphElement>('#status')!
const vrButton = document.querySelector<HTMLButtonElement>('#enter-vr')!
const arButton = document.querySelector<HTMLButtonElement>('#enter-ar')!
const bpmInput = document.querySelector<HTMLInputElement>('#bpm')!
const bpmValue = document.querySelector<HTMLSpanElement>('#bpm-value')!
const colorInput = document.querySelector<HTMLInputElement>('#color')!
const canvas = document.querySelector<HTMLCanvasElement>('#preview')!

async function main() {
  if (!navigator.gpu) {
    status.textContent = 'WebGPU is not supported on this browser.'
    return
  }
  // XR で使うため xrCompatible なアダプターを要求する
  const adapter = await navigator.gpu.requestAdapter({ xrCompatible: true })
  if (!adapter) {
    status.textContent = 'WebGPU adapter is not available.'
    return
  }
  const device = await adapter.requestDevice()
  device.lost.then((info) => {
    status.textContent = `GPU device lost: ${info.message}`
  })

  const scene = new Scene({ bpm: 120, r: 255, g: 255, b: 255 })
  const renderer = new SceneRenderer(device, 4)
  const preview = new DesktopPreview(canvas, device, renderer, scene)
  preview.start()

  bpmInput.addEventListener('input', () => {
    scene.params.bpm = Number(bpmInput.value)
    bpmValue.textContent = bpmInput.value
  })
  colorInput.addEventListener('input', () => {
    const hex = colorInput.value
    scene.params.r = Number.parseInt(hex.slice(1, 3), 16)
    scene.params.g = Number.parseInt(hex.slice(3, 5), 16)
    scene.params.b = Number.parseInt(hex.slice(5, 7), 16)
  })

  let session: XRSession | null = null

  const setup = async (button: HTMLButtonElement, mode: XRMode, label: string) => {
    const supported = isXRGPUBindingSupported() && await isXRModeSupported(mode)
    button.disabled = !supported
    button.title = supported ? '' : `${mode} (WebGPU) is not supported`
    button.addEventListener('click', async () => {
      if (session) {
        await session.end()
        return
      }
      try {
        preview.stop()
        session = await startXRSession(mode, device, renderer, scene, () => {
          session = null
          vrButton.textContent = 'Enter VR'
          arButton.textContent = 'Enter AR'
          preview.resetField()
          preview.start()
        })
        button.textContent = `Exit ${label}`
      }
      catch (error) {
        console.error(error)
        status.textContent = `Failed to start ${label}: ${(error as Error).message}`
        preview.start()
      }
    })
    return supported
  }

  const [vr, ar] = await Promise.all([
    setup(vrButton, 'immersive-vr', 'VR'),
    setup(arButton, 'immersive-ar', 'AR'),
  ])
  if (!navigator.xr)
    status.textContent = 'WebXR is not available (HTTPS が必要です)。デスクトッププレビューで表示しています。'
  else if (!isXRGPUBindingSupported())
    status.textContent = 'WebXR/WebGPU binding (XRGPUBinding) が未対応です。デスクトッププレビューで表示しています。'
  else if (!vr && !ar)
    status.textContent = 'immersive-vr / immersive-ar に対応したデバイスが見つかりません。'
}

main().catch((error) => {
  console.error(error)
  status.textContent = (error as Error).message
})
