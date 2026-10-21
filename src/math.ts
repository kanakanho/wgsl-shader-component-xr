// 列優先 (column-major) の 4x4 行列ユーティリティ
// WebXR の XRRigidTransform.matrix / projectionMatrix と同じレイアウト

export type Mat4 = Float32Array
export type Vec3 = [number, number, number]

export function identity(): Mat4 {
  const m = new Float32Array(16)
  m[0] = m[5] = m[10] = m[15] = 1
  return m
}

export function multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16)
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0
      for (let k = 0; k < 4; k++)
        sum += a[k * 4 + row] * b[col * 4 + k]
      out[col * 4 + row] = sum
    }
  }
  return out
}

export function invert(m: Mat4): Mat4 {
  const [
    a00,
    a01,
    a02,
    a03,
    a10,
    a11,
    a12,
    a13,
    a20,
    a21,
    a22,
    a23,
    a30,
    a31,
    a32,
    a33,
  ] = m
  const b00 = a00 * a11 - a01 * a10
  const b01 = a00 * a12 - a02 * a10
  const b02 = a00 * a13 - a03 * a10
  const b03 = a01 * a12 - a02 * a11
  const b04 = a01 * a13 - a03 * a11
  const b05 = a02 * a13 - a03 * a12
  const b06 = a20 * a31 - a21 * a30
  const b07 = a20 * a32 - a22 * a30
  const b08 = a20 * a33 - a23 * a30
  const b09 = a21 * a32 - a22 * a31
  const b10 = a21 * a33 - a23 * a31
  const b11 = a22 * a33 - a23 * a32
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06
  if (!det)
    return identity()
  det = 1 / det
  return new Float32Array([
    (a11 * b11 - a12 * b10 + a13 * b09) * det,
    (a02 * b10 - a01 * b11 - a03 * b09) * det,
    (a31 * b05 - a32 * b04 + a33 * b03) * det,
    (a22 * b04 - a21 * b05 - a23 * b03) * det,
    (a12 * b08 - a10 * b11 - a13 * b07) * det,
    (a00 * b11 - a02 * b08 + a03 * b07) * det,
    (a32 * b02 - a30 * b05 - a33 * b01) * det,
    (a20 * b05 - a22 * b02 + a23 * b01) * det,
    (a10 * b10 - a11 * b08 + a13 * b06) * det,
    (a01 * b08 - a00 * b10 - a03 * b06) * det,
    (a30 * b04 - a31 * b02 + a33 * b00) * det,
    (a21 * b02 - a20 * b04 - a23 * b00) * det,
    (a11 * b07 - a10 * b09 - a12 * b06) * det,
    (a00 * b09 - a01 * b07 + a02 * b06) * det,
    (a31 * b01 - a30 * b03 - a32 * b00) * det,
    (a20 * b03 - a21 * b01 + a22 * b00) * det,
  ])
}

export function translation(x: number, y: number, z: number): Mat4 {
  const m = identity()
  m[12] = x
  m[13] = y
  m[14] = z
  return m
}

export function scaling(x: number, y: number, z: number): Mat4 {
  const m = identity()
  m[0] = x
  m[5] = y
  m[10] = z
  return m
}

// WebGPU のクリップ空間 (depth 0..1) 用の透視投影行列
export function perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2)
  const m = new Float32Array(16)
  m[0] = f / aspect
  m[5] = f
  m[10] = far / (near - far)
  m[11] = -1
  m[14] = (near * far) / (near - far)
  return m
}

export function getPosition(m: Mat4): Vec3 {
  return [m[12], m[13], m[14]]
}

export function transformPoint(m: Mat4, p: Vec3): Vec3 {
  return [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
    m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
    m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
  ]
}

export function transformDir(m: Mat4, d: Vec3): Vec3 {
  return [
    m[0] * d[0] + m[4] * d[1] + m[8] * d[2],
    m[1] * d[0] + m[5] * d[1] + m[9] * d[2],
    m[2] * d[0] + m[6] * d[1] + m[10] * d[2],
  ]
}

export function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]) || 1
  return [v[0] / len, v[1] / len, v[2] / len]
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

// origin に位置し、-Z 軸が dir を向く姿勢行列 (XR の targetRaySpace と同じ規約)
export function poseFromRay(origin: Vec3, dir: Vec3): Mat4 {
  const z = normalize([-dir[0], -dir[1], -dir[2]])
  let x = cross([0, 1, 0], z)
  if (Math.hypot(...x) < 1e-6)
    x = [1, 0, 0]
  x = normalize(x)
  const y = cross(z, x)
  return new Float32Array([
    x[0],
    x[1],
    x[2],
    0,
    y[0],
    y[1],
    y[2],
    0,
    z[0],
    z[1],
    z[2],
    0,
    origin[0],
    origin[1],
    origin[2],
    1,
  ])
}
