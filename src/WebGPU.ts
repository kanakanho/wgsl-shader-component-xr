import WebGPURender from './WebGPURender'

export interface WebGPUFragment {
  bpm: number
  r: number
  g: number
  b: number
}

interface WebGPUProps extends WebGPUFragment {
  shaderSource: string
}

// 2D (フルスクリーン) 描画。React 非依存のため canvas を直接受け取り、停止関数を返す
export default function WebGPU(canvas: HTMLCanvasElement, { bpm, r, g, b, shaderSource }: WebGPUProps) {
  let isDisposed = false
  let stopRender: (() => void) | null = null

  const fragValue: WebGPUFragment = {
    bpm,
    r,
    g,
    b,
  }

  const initWebGPU = async () => {
    // webgpuコンテキストの取得
    const context = canvas.getContext('webgpu') as GPUCanvasContext

    // deviceの取得
    const g_adapter = await navigator.gpu.requestAdapter()
    if (!g_adapter) {
      console.error('WebGPU is not supported on this browser.')
      return
    }
    const g_device = await g_adapter.requestDevice()
    if (isDisposed)
      return

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat()
    context.configure({
      device: g_device,
      format: presentationFormat,
      alphaMode: 'opaque', // or 'premultiplied'
    })

    stopRender = WebGPURender(
      canvas,
      g_device,
      context,
      presentationFormat,
      4,
      shaderSource,
      fragValue,
    )
  }

  initWebGPU()
  return () => {
    isDisposed = true
    if (stopRender)
      stopRender()
  }
}
