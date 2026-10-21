// XR 空間描画用シェーダー (六角形グリッドの 3D 拡張)
//
// 2D 版の構成:
//   - 2 つの矩形格子 a, b をずらして重ね、近い方の中心を選ぶ -> 六角形タイル (六方格子のボロノイ)
//   - sdHexagon で六角形の輪郭までの距離 d を求め、pow(0.04 / |d|, 1.2) で輪郭を光らせる
//   - 少し小さい六角形 (bh * 0.87) を描くので、隣のセルとの間に二重線ができる
//
// 3D 版の構成:
//   - 立方格子と (0.5, 0.5, 0.5) ずらした立方格子を重ね、近い方を選ぶ -> 体心立方格子のボロノイ
//     = 切頂八面体 (六角形 8 面 + 正方形 6 面) が空間を隙間なく埋めるハニカム
//   - 面の平面までの距離から、辺 (稜線) と面までの距離を求めて光らせる
//   - 2D 版と同じく少し縮めた多面体を描くので、隣のセルとの間に二重の稜線ができる
//
// バインディング・エントリーポイント (vs_space / fs_space / vs_cube / fs_cube) は XR 版と同じ

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

const MAX_STEPS = 160;
const MAX_DIST = 12.0;

const CELL_SIZE = 0.6;        // セル (立方格子) の一辺 (フィールド単位)
const SHRINK_GAP = 0.04;      // 多面体を縮める量 (セル単位)。2D 版の bh * 0.87 に相当
const INV_SQRT3 = 0.577350269;

const EDGE_WIDTH = 0.008;     // 稜線の太さ (フィールド単位)
const EDGE_GAIN = 45.0;
const HALO_RADIUS = 0.02;     // 2D 版の pow(0.04 / d, 1.2) に相当する裾
const HALO_GAIN = 0.6;
const FACE_WIDTH = 0.004;     // 面の厚さ (フィールド単位)
const FACE_GAIN = 3.0;

fn quad_vertex(index: u32) -> vec2f {
  var pos = array<vec2f, 6>(
    vec2f(-1.0,  1.0), vec2f(-1.0, -1.0), vec2f( 1.0, -1.0),
    vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0)
  );
  return pos[index];
}

// 2D 版と同じパレット
fn colorPalette(t: f32) -> vec3<f32> {
  let a = vec3(0.5, 0.5, 0.5);
  let b = vec3(0.5, 0.5, 0.5);
  let c = vec3(1.0, 1.0, 1.0);
  let d = vec3(0.50, 0.20, 0.10);
  return a + b * cos(6.28318*(c*t+d));
}

fn hash3(p: vec3f) -> f32 {
  return fract(sin(dot(p, vec3f(127.1, 311.7, 74.7))) * 43758.5453);
}

// ---------------------------------------------------------------
// 切頂八面体ハニカム
// ---------------------------------------------------------------

struct Honeycomb {
  local: vec3f,   // セル中心からの相対座標 (セル単位)
  center: vec3f,  // セル中心 (セル単位)
  edge: f32,      // 縮めた多面体の稜線までの距離 (セル単位, 近似)
  face: f32,      // 縮めた多面体の面までの距離 (セル単位)
};

// 2 番目までの最小値を更新する
fn push_min(m: vec2f, v: f32) -> vec2f {
  if (v < m.x) {
    return vec2f(v, m.x);
  }
  return vec2f(m.x, min(m.y, v));
}

fn honeycomb(p: vec3f) -> Honeycomb {
  // 2D 版の a, b と同じ: ずらした 2 つの格子のうち近い方の中心を選ぶ
  let a = fract(p) - 0.5;
  let b = fract(p - 0.5) - 0.5;
  let gv = select(b, a, dot(a, a) < dot(b, b));
  let q = abs(gv);

  // 縮めた多面体の各面までの距離 (内側が正)
  //   正方形の面: |x| = 0.5
  //   六角形の面: |x| + |y| + |z| = 0.75 (abs で折り畳んだ象限の面と、隣の象限の 3 面)
  let s = q.x + q.y + q.z;
  var m = vec2f(1e3);
  m = push_min(m, 0.5 - q.x - SHRINK_GAP);
  m = push_min(m, 0.5 - q.y - SHRINK_GAP);
  m = push_min(m, 0.5 - q.z - SHRINK_GAP);
  m = push_min(m, (0.75 - s) * INV_SQRT3 - SHRINK_GAP);
  m = push_min(m, (0.75 - s + 2.0 * q.x) * INV_SQRT3 - SHRINK_GAP);
  m = push_min(m, (0.75 - s + 2.0 * q.y) * INV_SQRT3 - SHRINK_GAP);
  m = push_min(m, (0.75 - s + 2.0 * q.z) * INV_SQRT3 - SHRINK_GAP);

  var h: Honeycomb;
  h.local = gv;
  h.center = p - gv;
  // 稜線 = 2 つの面が交わる線。最も近い 2 面までの距離から求める
  h.edge = length(m);
  h.face = abs(m.x);
  return h;
}

// ---------------------------------------------------------------
// 3D フィールド
// ---------------------------------------------------------------

struct FieldSample {
  dist: f32,        // 発光する面・稜線までの距離 (ステップ幅の決定に使用)
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

  let r0 = length(p);
  let h = honeycomb(p / CELL_SIZE);
  let edge_d = h.edge * CELL_SIZE;
  let face_d = h.face * CELL_SIZE;

  var res: FieldSample;
  res.dist = face_d;

  // 2D 版: colorPalette(length(uv0) + time * 0.5)。セルごとに少し色をずらす
  let cell_rand = hash3(h.center);
  let col = colorPalette(r0 + u.time * 0.5 + cell_rand * 0.15);

  // 稜線: 細いガウスの芯 + 2D 版の pow(0.04 / d, 1.2) に似た短い裾
  //   幅を広げたときは明るさを下げ、線を横切ったときの総量をほぼ一定に保つ
  let ew = max(EDGE_WIDTH, width);
  let core = exp(-edge_d * edge_d / (ew * ew)) * (EDGE_WIDTH / ew);
  let halo_t = HALO_RADIUS / (edge_d + HALO_RADIUS);
  let halo = halo_t * halo_t * HALO_GAIN;
  let edge_glow = core * EDGE_GAIN + halo;

  // 面: ごく薄く光らせて六角形・正方形の面を見せる
  let fw = max(FACE_WIDTH, width);
  let face_glow = exp(-face_d * face_d / (fw * fw)) * (FACE_WIDTH / fw) * FACE_GAIN;

  // 中心から外へ進む明るさの波と、中心から離れるほどの減衰
  let wave = 0.35 + 0.65 * pow(0.5 + 0.5 * sin(r0 * 3.0 - u.time * 1.5), 4.0);
  let envelope = exp(-r0 * 0.45) * (0.6 + 0.4 * cell_rand);

  res.emission = col * (edge_glow * wave + face_glow) * envelope;
  res.density = (core * EDGE_GAIN * 0.2 + face_glow * 0.5) * envelope;

  // コントローラー先端の光
  for (var i = 0; i < 2; i++) {
    let ptr = u.pointers[i];
    if (ptr.w > 0.5) {
      let c = (u.inv_model * vec4f(ptr.xyz, 1.0)).xyz;
      let dist = length(p_in - c);
      let pw = 0.03 * (1.0 + step(1.5, ptr.w));
      res.emission += vec3f(0.6, 0.85, 1.0) * 8.0 * pw * pw / (dist * dist + pw * pw);
    }
  }
  return res;
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
    let dt = clamp(s.dist * 0.8, min_step, 0.2);

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
