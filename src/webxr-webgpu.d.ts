// WebXR/WebGPU Binding の型定義
// https://github.com/immersive-web/WebXR-WebGPU-Binding/blob/main/explainer.md
// @types/webxr には未収録のため最低限の宣言を追加する

interface GPURequestAdapterOptions {
  xrCompatible?: boolean
}

interface XRGPUProjectionLayerInit {
  colorFormat: GPUTextureFormat
  depthStencilFormat?: GPUTextureFormat
  textureUsage?: GPUTextureUsageFlags
  scaleFactor?: number
}

interface XRGPUSubImage extends XRSubImage {
  readonly colorTexture: GPUTexture
  readonly depthStencilTexture?: GPUTexture | null
  getViewDescriptor: () => GPUTextureViewDescriptor
}

declare class XRGPUBinding {
  constructor(session: XRSession, device: GPUDevice)
  getPreferredColorFormat(): GPUTextureFormat
  createProjectionLayer(init: XRGPUProjectionLayerInit): XRProjectionLayer
  getViewSubImage(layer: XRProjectionLayer, view: XRView): XRGPUSubImage
}

interface Window {
  XRGPUBinding?: typeof XRGPUBinding
}
