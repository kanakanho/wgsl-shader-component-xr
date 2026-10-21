import type { Mat4, Vec3 } from './math'
import { invert } from './math'
import shaderSource from './shader/xr-hexagram.wgsl?raw'

export const MAX_INSTANCES = 32
const INSTANCE_FLOATS = 20 // mat4 + vec4
const SCENE_FLOATS = 32 // mat4 + 4 + 4 + 2 * vec4
const CAMERA_FLOATS = 36 // mat4 * 2 + vec4

export interface RenderView {
  viewProj: Mat4
  cameraPosition: Vec3
  viewport?: { x: number, y: number, width: number, height: number }
}

// 1 つのテクスチャ (配列レイヤー) に対する描画。
// 左右の目が同じテクスチャを viewport で分割している場合は views が複数になる
export interface RenderTarget {
  texture: GPUTexture
  viewDescriptor?: GPUTextureViewDescriptor
  views: RenderView[]
}

export interface FrameData {
  sceneUniform: Float32Array
  instanceData: Float32Array
  instanceCount: number
  clearColor: GPUColor
}

interface Pipelines {
  space: GPURenderPipeline
  cube: GPURenderPipeline
}

export default class SceneRenderer {
  private device: GPUDevice
  private sampleCount: number
  private shaderModule: GPUShaderModule
  private pipelineLayout: GPUPipelineLayout
  private cameraLayout: GPUBindGroupLayout
  private pipelines = new Map<GPUTextureFormat, Pipelines>()

  private sceneBuffer: GPUBuffer
  private instanceBuffer: GPUBuffer
  private sceneBindGroup: GPUBindGroup
  // 1 フレーム内の各ビューごとにカメラ用バッファを用意する
  // (writeBuffer は submit 時にまとめて反映されるため共有できない)
  private cameras: { buffer: GPUBuffer, bindGroup: GPUBindGroup }[] = []

  private attachmentKey = ''
  private msaaTexture: GPUTexture | null = null
  private depthTexture: GPUTexture | null = null

  constructor(device: GPUDevice, sampleCount: number = 4) {
    this.device = device
    this.sampleCount = sampleCount
    this.shaderModule = device.createShaderModule({ code: shaderSource })

    this.cameraLayout = device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      }],
    })
    const sceneLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'read-only-storage' },
        },
      ],
    })
    this.pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.cameraLayout, sceneLayout],
    })

    this.sceneBuffer = device.createBuffer({
      size: SCENE_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.instanceBuffer = device.createBuffer({
      size: MAX_INSTANCES * INSTANCE_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.sceneBindGroup = device.createBindGroup({
      layout: sceneLayout,
      entries: [
        { binding: 0, resource: { buffer: this.sceneBuffer } },
        { binding: 1, resource: { buffer: this.instanceBuffer } },
      ],
    })
  }

  private getPipelines(format: GPUTextureFormat): Pipelines {
    const cached = this.pipelines.get(format)
    if (cached)
      return cached

    // premultiplied alpha ブレンド (AR で背景を透過させるため)
    const blend: GPUBlendState = {
      color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    }
    const create = (vs: string, fs: string, depthWriteEnabled: boolean, depthCompare: GPUCompareFunction) =>
      this.device.createRenderPipeline({
        layout: this.pipelineLayout,
        vertex: { module: this.shaderModule, entryPoint: vs },
        fragment: {
          module: this.shaderModule,
          entryPoint: fs,
          targets: [{ format, blend }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: {
          format: 'depth24plus',
          depthWriteEnabled,
          depthCompare,
        },
        multisample: { count: this.sampleCount },
      })

    const pipelines: Pipelines = {
      // 空間全体のレイマーチング (フルスクリーン、深度は書かない)
      space: create('vs_space', 'fs_space', false, 'always'),
      cube: create('vs_cube', 'fs_cube', true, 'less'),
    }
    this.pipelines.set(format, pipelines)
    return pipelines
  }

  private getCamera(index: number) {
    while (this.cameras.length <= index) {
      const buffer = this.device.createBuffer({
        size: CAMERA_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      const bindGroup = this.device.createBindGroup({
        layout: this.cameraLayout,
        entries: [{ binding: 0, resource: { buffer } }],
      })
      this.cameras.push({ buffer, bindGroup })
    }
    return this.cameras[index]
  }

  // MSAA と深度バッファをターゲットのサイズに合わせて (再) 作成する
  private getAttachments(texture: GPUTexture) {
    const key = `${texture.width}x${texture.height}:${texture.format}`
    if (key !== this.attachmentKey || !this.depthTexture) {
      this.msaaTexture?.destroy()
      this.depthTexture?.destroy()
      const size = [texture.width, texture.height]
      this.msaaTexture = this.sampleCount > 1
        ? this.device.createTexture({
            size,
            format: texture.format,
            sampleCount: this.sampleCount,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
          })
        : null
      this.depthTexture = this.device.createTexture({
        size,
        format: 'depth24plus',
        sampleCount: this.sampleCount,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })
      this.attachmentKey = key
    }
    return {
      msaaView: this.msaaTexture?.createView() ?? null,
      depthView: this.depthTexture.createView(),
    }
  }

  render(targets: RenderTarget[], frame: FrameData) {
    const { device } = this
    device.queue.writeBuffer(this.sceneBuffer, 0, frame.sceneUniform)
    if (frame.instanceCount > 0) {
      device.queue.writeBuffer(
        this.instanceBuffer,
        0,
        frame.instanceData,
        0,
        frame.instanceCount * INSTANCE_FLOATS,
      )
    }

    const commandEncoder = device.createCommandEncoder()
    let cameraIndex = 0

    for (const target of targets) {
      const pipelines = this.getPipelines(target.texture.format)
      const { msaaView, depthView } = this.getAttachments(target.texture)
      const targetView = target.texture.createView(target.viewDescriptor)

      const passEncoder = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: msaaView ?? targetView,
          resolveTarget: msaaView ? targetView : undefined,
          clearValue: frame.clearColor,
          loadOp: 'clear',
          storeOp: 'store',
        }],
        depthStencilAttachment: {
          view: depthView,
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      })
      passEncoder.setBindGroup(1, this.sceneBindGroup)

      for (const view of target.views) {
        const camera = this.getCamera(cameraIndex++)
        const cameraData = new Float32Array(CAMERA_FLOATS)
        cameraData.set(view.viewProj, 0)
        cameraData.set(invert(view.viewProj), 16)
        cameraData.set(view.cameraPosition, 32)
        cameraData[35] = 1
        device.queue.writeBuffer(camera.buffer, 0, cameraData)

        if (view.viewport) {
          const { x, y, width, height } = view.viewport
          passEncoder.setViewport(x, y, width, height, 0, 1)
          passEncoder.setScissorRect(x, y, width, height)
        }
        passEncoder.setBindGroup(0, camera.bindGroup)

        passEncoder.setPipeline(pipelines.space)
        passEncoder.draw(6)
        if (frame.instanceCount > 0) {
          passEncoder.setPipeline(pipelines.cube)
          passEncoder.draw(36, frame.instanceCount)
        }
      }
      passEncoder.end()
    }

    device.queue.submit([commandEncoder.finish()])
  }
}
