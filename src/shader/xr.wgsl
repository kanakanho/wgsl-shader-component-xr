// XR 空間描画用シェーダー
// base.wgsl のフラクタル (fract による空間の折り畳み + sin の同心殻) を 3 次元に拡張し、
// 各ピクセルの視線レイで SDF をレイマーチングして空間全体のボリュームとして描画する

struct Camera {
  view_proj: mat4x4f,
  inv_view_proj: mat4x4f,
  position: vec4f,
};

struct Scene {
  inv_model: mat4x4f,        // ワールド座標 -> フィールドのローカル座標
  time: f32,
  field_scale: f32,          // フィールド 1 単位あたりのメートル数
  floor_y: f32,
  ar_mode: f32,              // 1.0 のとき暗部を透過させる
  color: vec3f,
  floor_enabled: f32,
  pointers: array<vec4f, 2>, // xyz: コントローラー先端のワールド座標, w: 0=なし 1=あり 2=押下中
};

struct Instance {
  model: mat4x4f,
  color: vec4f,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> u: Scene;
@group(1) @binding(1) var<storage, read> instances: array<Instance>;

const ITERATIONS = 5;
const MAX_STEPS = 128;
const MAX_DIST = 12.0;

fn quad_vertex(index: u32) -> vec2f {
  var pos = array<vec2f, 6>(
    vec2f(-1.0,  1.0), vec2f(-1.0, -1.0), vec2f( 1.0, -1.0),
    vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0)
  );
  return pos[index];
}

fn colorPalette(t: f32) -> vec3<f32> {
  let a = vec3(0.5, 0.5, 0.5);
  let b = vec3(0.5, 0.5, 0.5);
  let c = vec3(1.0, 1.0, 1.0);
  let d = vec3(0.00, 0.10, 0.20);
  return a + b * cos(6.28318*(c*t+d));
}

// ---------------------------------------------------------------
// 3D SDF フィールド
// ---------------------------------------------------------------

struct FieldSample {
  dist: f32,        // 最も粗い殻までの距離 (ステップ幅の決定に使用)
  emission: vec3f,  // 単位長さあたりの発光
  density: f32,     // 単位長さあたりの吸収
};

// p: フィールドのローカル座標, width: 発光の幅 (遠いほど太くしてエイリアスを抑える)
fn field(p_in: vec3f, width: f32) -> FieldSample {
  var p = p_in;

  // コントローラーの位置で空間を歪ませる
  for (var i = 0; i < 2; i++) {
    let ptr = u.pointers[i];
    if (ptr.w > 0.5) {
      let c = (u.inv_model * vec4f(ptr.xyz, 1.0)).xyz;
      let delta = p - c;
      let dist = length(delta);
      let strength = (0.15 + step(1.5, ptr.w) * 0.35) * exp(-dist * 3.0);
      p += delta * strength * sin(dist * 12.0 - u.time * 4.0);
    }
  }

  let p0 = p;
  let r0 = length(p0);
  let fade = exp(-r0);

  var q = p;
  var scale = 1.0;
  var out: FieldSample;
  out.dist = 1e3;
  out.emission = vec3f(0.0);
  out.density = 0.0;

  let inv_w2 = 1.0 / (width * width);
  for (var i = 0; i < ITERATIONS; i++) {
    // base.wgsl と同じ折り畳み: uv = fract(uv * 1.5) - 0.5 を 3 次元で行う
    q = fract(q * 1.5) - 0.5;
    scale *= 1.5;

    // base.wgsl: d = sin(length(uv) * exp(-length(uv0)) * 8 + time) の 0 となる殻
    let s = sin(length(q) * fade * 8.0 + u.time);
    // 殻までの距離の近似 (勾配 = 8 * fade * scale で割る)
    let d = abs(s) / (8.0 * max(fade, 1e-3) * scale);
    out.dist = min(out.dist, d);

    // 殻の近くだけ光らせる (ガウス: 裾を持たないので空間が霞まない)
    let col = colorPalette(r0 + f32(i) * 0.4 + u.time * 0.4);
    let glow = exp(-d * d * inv_w2) / (1.0 + f32(i) * 0.6);
    out.emission += col * glow;
    out.density += glow;
  }

  // 中心から離れるほど減衰 (base.wgsl の exp(-length(uv0)) に相当)
  let envelope = fade * 15.0;
  out.emission *= envelope;
  out.density *= envelope * 0.3;

  // コントローラー先端の光
  for (var i = 0; i < 2; i++) {
    let ptr = u.pointers[i];
    if (ptr.w > 0.5) {
      let c = (u.inv_model * vec4f(ptr.xyz, 1.0)).xyz;
      let dist = length(p_in - c);
      let pw = 0.03 * (1.0 + step(1.5, ptr.w));
      out.emission += vec3f(0.6, 0.85, 1.0) * 8.0 * pw * pw / (dist * dist + pw * pw);
    }
  }
  return out;
}

// ---------------------------------------------------------------
// 空間全体のレイマーチング (フルスクリーン)
// ---------------------------------------------------------------

struct SpaceOutput {
  @builtin(position) position: vec4f,
  @location(0) ndc: vec2f,
};

@vertex
fn vs_space(@builtin(vertex_index) VertexIndex: u32) -> SpaceOutput {
  let p = quad_vertex(VertexIndex);
  var output: SpaceOutput;
  output.position = vec4f(p, 0.0, 1.0);
  output.ndc = p;
  return output;
}

fn unproject(ndc: vec2f, z: f32) -> vec3f {
  let h = camera.inv_view_proj * vec4f(ndc, z, 1.0);
  return h.xyz / h.w;
}

fn floor_grid(p: vec2f, t: f32) -> f32 {
  let g = abs(fract(p * 2.0 - 0.5) - 0.5);
  let w = 0.01 + t * 0.003;
  return 1.0 - smoothstep(0.0, w * 2.0, min(g.x, g.y));
}

@fragment
fn fs_space(input: SpaceOutput) -> @location(0) vec4f {
  // 視線レイ (ワールド座標)
  let ro_w = unproject(input.ndc, 0.0);
  let rd_w = normalize(unproject(input.ndc, 0.5) - ro_w);

  // 床との交差 (ワールド距離)
  var t_floor = 1e9;
  if (u.floor_enabled > 0.5 && rd_w.y < -1e-4) {
    t_floor = (u.floor_y - ro_w.y) / rd_w.y;
  }

  // フィールドのローカル座標でマーチングする (一様スケールなので距離は field_scale 倍)
  let ro = (u.inv_model * vec4f(ro_w, 1.0)).xyz;
  let rd = normalize((u.inv_model * vec4f(rd_w, 0.0)).xyz);
  let inv_scale = 1.0 / u.field_scale;
  let t_max = min(MAX_DIST, t_floor * inv_scale);

  var t = 0.02 * inv_scale;
  var transmittance = 1.0;
  var color = vec3f(0.0);

  for (var i = 0; i < MAX_STEPS; i++) {
    if (t > t_max || transmittance < 0.02) {
      break;
    }
    let min_step = 0.004 + t * 0.008;
    let width = min_step * 0.9;
    let s = field(ro + rd * t, width);
    let dt = clamp(s.dist * 0.7, min_step, 0.25);

    color += transmittance * s.emission * dt;
    transmittance *= exp(-s.density * dt);
    t += dt;
  }

  // 白飛びを抑えるトーンマッピング
  color = 1.0 - exp(-color * u.color * 1.5);

  // 床グリッド
  if (t_floor < 1e8 && t >= t_max && t_max < MAX_DIST) {
    let hit = ro_w + rd_w * t_floor;
    let line = floor_grid(hit.xz, t_floor);
    let fade = exp(-length(hit.xz - camera.position.xz) * 0.25);
    color += transmittance * vec3f(0.25, 0.35, 0.5) * line * fade;
  }

  if (u.ar_mode > 0.5) {
    // AR では暗部を透過 (premultiplied alpha)
    let c = clamp(color, vec3f(0.0), vec3f(1.0));
    let a = max(max(c.r, max(c.g, c.b)), 1.0 - transmittance);
    return vec4f(c, a);
  }
  return vec4f(color + transmittance * vec3f(0.01, 0.01, 0.02), 1.0);
}

// ---------------------------------------------------------------
// コントローラー (インスタンス描画のキューブ)
// ---------------------------------------------------------------

struct CubeOutput {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
  @location(1) color: vec4f,
};

@vertex
fn vs_cube(
  @builtin(vertex_index) VertexIndex: u32,
  @builtin(instance_index) InstanceIndex: u32,
) -> CubeOutput {
  var normals = array<vec3f, 6>(
    vec3f( 1.0, 0.0, 0.0), vec3f(-1.0, 0.0, 0.0),
    vec3f( 0.0, 1.0, 0.0), vec3f( 0.0,-1.0, 0.0),
    vec3f( 0.0, 0.0, 1.0), vec3f( 0.0, 0.0,-1.0)
  );
  let n = normals[VertexIndex / 6u];
  let t = vec3f(n.y, n.z, n.x);
  let b = cross(n, t);
  let c = quad_vertex(VertexIndex % 6u);
  let local = (n + t * c.x + b * c.y) * 0.5;

  let inst = instances[InstanceIndex];
  var output: CubeOutput;
  output.position = camera.view_proj * inst.model * vec4f(local, 1.0);
  output.normal = normalize((inst.model * vec4f(n, 0.0)).xyz);
  output.color = inst.color;
  return output;
}

@fragment
fn fs_cube(input: CubeOutput) -> @location(0) vec4f {
  let light = 0.45 + 0.55 * max(dot(normalize(input.normal), normalize(vec3f(0.3, 1.0, 0.5))), 0.0);
  return vec4f(input.color.rgb * light * input.color.a, input.color.a);
}
