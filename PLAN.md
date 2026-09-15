# GPU Boids 3D (50k–100k агентов) на чистом Rust + wgpu

Технический документ и план реализации. Спринт: 4 дня плотной разработки.
Статус: Day 1 — in progress.

## 1. Цель

Полностью GPU-резидентная симуляция роя (Boids + spatial grid + bitonic sort в WGSL compute),
два режима (подводный биолюминесцентный / небесный с процедурными биомами),
raycast-интерактив мышью (аттрактор/репеллер), SDF-огибание препятствий, HDR-постобработка.
CPU за кадр: только запись uniform-буферов и command buffer. Никаких циклов по агентам на CPU.

Целевое железо: Linux + NVIDIA (RTX 4060 Ti, Vulkan). Разрешение адаптивное 1920x1080 .. 3840x2160,
основной таргет 1440p. Все проходы масштабируются от размера swapchain, никаких хардкод-размеров.

## 2. Cargo Workspace: 5 крейтов

```
swarm_simulation/
  Cargo.toml                 # [workspace] members, workspace deps, profile
  PLAN.md                    # этот документ
  docs/                      # ADR, math.md, gpu-pipeline.md, perf.md
  crates/
    boids-core/              # контракт данных + CPU-эталон. Без wgpu.
    boids-gpu/               # буферы, bind groups, compute pipelines, диспетчи
    boids-scene/             # SDF-поля, heightmap/biomes, небо, палитры, меши
    boids-render/            # проходы рендера: instanced boids, окружение, bloom, tonemap
    boids-app/               # runtime: winit, surface, ввод, камера, HUD
```

### boids-core
Ответственность: единственный источник правды о layout'ах и математике; эталон для валидации GPU.

* `#[repr(C, align(16))] Boid` — 48 B: `pos: [f32;3]`, `species: f32`, `vel: [f32;3]`, `phase: f32`,
  `color_seed: f32`, `energy: f32`, `prev_dir_x/z: [f32;2]` (для bank-угла в VS).
* `KeyVal { key: u32, val: u32 }` — 8 B, ключ сортировки = linear cell index.
* `SimParams` (128 B), `InteractionUniforms` (48 B), `CameraUniform`, `SortParams` — все align(16).
* `const _: () = assert!(...)` на каждый размер/выравнивание. Ручной `unsafe impl Pod` запрещён.
* `reference::step_cpu(&mut [Boid], &SimParams)` — наивный O(N²) эталон.
* `sdf::{sphere, box, torus, column_field, plane, smin}` — те же формулы, что в WGSL.
* `camera::{OrbitCamera, ray_from_ndc, ray_plane_intersect}`, `math::{clamp_len, order_parameter}`.

deps: `bytemuck` (derive), `glam`. dev: `approx`.

### boids-gpu
Ответственность: движок вычислений. Ничего не знает о биомах, знает о контракте boids-core.

```rust
pub struct SimResources {          // все буферы, создаются один раз
    pub boids: [wgpu::Buffer; 2],  // ping-pong, STORAGE | COPY_SRC | COPY_DST
    pub keys: [wgpu::Buffer; 2],
    pub cell_start: wgpu::Buffer,
    pub cell_end: wgpu::Buffer,
    pub params: wgpu::Buffer,      // UNIFORM
    pub interaction: wgpu::Buffer, // UNIFORM
    pub sort_params: wgpu::Buffer, // UNIFORM, dynamic offset per stage
    pub read: usize,               // индекс актуального буфера
}
pub struct SimPipelines { hash, clear_cells, sort_local, sort_global, build_ranges,
                          integrate_fish, integrate_bird }
pub trait GpuStage { fn record(&self, enc: &mut CommandEncoder, res: &SimResources, ctx: &FrameCtx); }
pub fn record_frame(...) -> usize;   // возвращает индекс буфера с готовыми позициями
pub struct ShaderCache;              // include-препроцессор + опциональный hot-reload
```

deps: `wgpu`, `bytemuck`, `boids-core`, `log`.

### boids-scene
Ответственность: процедурное окружение и арт-контент.

```rust
pub trait Biome {
    fn kind(&self) -> BiomeKind;
    fn derive_wgsl(&self) -> &'static str;          // #include-фрагмент для shader-композиции
    fn env_bind_group(&self) -> &wgpu::BindGroup;
    fn palette(&self) -> PaletteUniform;
}
pub struct OceanBiome;   // SDF-колонны/арки, каустика, god rays
pub struct SkyBiome;     // Biome { Forest, Dunes, Canyon }, heightfield + вариант рассеяния
pub struct HeightfieldGpu;  // RG32Float (height, biome_mask), генерится compute-проходом
pub mod mesh;               // fish_lowpoly(), bird_lowpoly(), tree_lowpoly()
```

deps: `wgpu`, `bytemuck`, `glam`, `boids-core`.

### boids-render
Ответственность: граф проходов кадра.

```rust
pub struct RenderCtx { device, queue, targets: HdrTargets, depth, camera, frame }
pub trait RenderMode { fn record(&mut self, ctx: &mut RenderCtx); fn resize(&mut self, size); }
pub struct FishMode; pub struct BirdMode;      // swap активного режима без пересоздания device
pub struct PostChain;                          // bloom, tonemap ACES, CA, vignette, grain
pub struct HdrTargets;                         // Rgba16Float, пересоздаётся на resize
```

deps: `wgpu`, `bytemuck`, `glam`, `boids-core`, `boids-scene`.

### boids-app
Ответственность: окно, ввод, камера, HUD, оркестрация, переключение режимов.

deps: `wgpu`, `winit`, `glam`, `pollster`, `env_logger`, `log`, `boids-*`.

## 3. GPU pipeline и data layout

### 3.1 Выравнивание (std430 / std140)

Все буферные структуры кратны 16 байт. `vec3<f32>` в WGSL имеет выравнивание 16, поэтому в Rust
никогда не ставим `[f32;3]` без соседнего скаляра, закрывающего дырку.

| Rust | size | WGSL | ключевые правила |
|---|---|---|---|
| `Boid` | 48 | `struct Boid` | array<Boid> stride 48, ок при 16-выравнивании полей |
| `KeyVal` | 8 | `struct KeyVal` | только u32 |
| `SimParams` | 128 | `struct SimParams` | каждый vec3 + f32/u32 |
| `InteractionUniforms` | 48 | `struct Interaction` | ray_origin+f32, focus+f32, ... |
| `SortParams` | 16 | `struct SortParams` | j, k, n, pad |

Тест `layout_matches_wgsl` пишет `offset_of`-значения из WGSL compute-прохода в буфер и сравнивает
с Rust — расхождение ловится в CI, а не в рантайме.

### 3.2 Буферы

| Буфер | Размер @100k | usage |
|---|---|---|
| `boids[2]` | 2 x 4.8 MB (48 B x 100k) | STORAGE, COPY_SRC, COPY_DST |
| `keys[2]` | 2 x 0.8 MB (padded до pow2) | STORAGE, COPY_SRC |
| `cell_start`, `cell_end` | 2 x 4.2 MB (128x64x128 u32) | STORAGE |
| `params`, `interaction`, `camera` | < 1 KB | UNIFORM, write_buffer раз в кадр |
| `sort_params` | 16 B x num_stages | UNIFORM с dynamic offset |

Домен ограничен `bounds_half`, сетка фиксированная 128x64x128 -> прямая индексация без хеша
(нет коллизий). Пустая ячейка = `U32_MAX` в `cell_start`.

### 3.3 Порядок compute-проходов кадра

```
P0 clear_cells   dispatch ceil(num_cells/256)   cell_start[i] = U32_MAX
P1 hash          dispatch ceil(n_padded/256)   keys[i] = KeyVal{cell_index(pos), i}; padding -> U32_MAX
P2 bitonic       sort_local (1 dispatch, tile 512 в workgroup) + sort_global (по dispatch на (k,j))
P3 build_ranges  if keys[i].key != keys[i-1].key { cell_start[k]=i; cell_end[prev]=i } + терминатор
P4 integrate     один поток на boid: 3x3x3 ячеек -> силы -> SDF -> курсор -> интеграция
```

Sort: N padded до 131072 -> 17*18/2 = 153 стадии. `sort_local` съедает стадии с `k <= 512`
внутри workgroup (shared memory tile), остальные идут глобальными dispatch'ами через dynamic offset.

### 3.4 Рендер boids

Один instanced draw call. Никакого vertex buffer: позиция из `@builtin(vertex_index)` в
процедурный меш, трансформ из `boids[read][@builtin(instance_index)]` (vertex pulling).

```wgsl
let fwd   = normalize(b.vel);
let up_r  = select(vec3f(0,1,0), vec3f(1,0,0), abs(fwd.y) > 0.99);
let right = normalize(cross(up_r, fwd));
let up    = cross(fwd, right);
let M     = mat3x3f(right, up, fwd);          // локальный меш смотрит в +Z
let bank  = clamp(dot(cross(normalize(b.prev_dir), fwd), up) * k_bank, -0.7, 0.7);
let wave  = sin(P.time * f + b.phase) * amp * smoothstep(0.0, L, local.z);
let world = b.pos + M * (local + vec3f(wave, 0, 0));   // + поворот bank вокруг fwd
```

## 4. Процедурные биомы и шейдинг

### Океан
* Окружение — fullscreen raymarch SDF (sphere tracing, <=64 шагов, adaptive eps), пишет `gl_FragDepth`,
  поэтому рыбы корректно перекрываются геометрией рифа.
* SDF: домен-репитиция колонн `d = length(vec2(mod(p.xz,R) - 0.5*R)) - r(p.y)`, `smin()` с полом,
  арки и навесы через вторую семью колонн.
* Каустика без текстур: два слоя анимированного Worley по (x,z) от проекции на дно,
  `caustic = pow(1-w1, 8) + pow(1-w2, 8)`, модулируется глубиной и нормалью.
* Beer-Lambert: `L = L0*exp(-sigma_e*d) + L_scatter*(1-exp(-sigma_s*d))`, `sigma_e = (0.45,0.12,0.06)`.
* God rays: 32 шага raymarch вдоль луча камеры с дешёвым SDF-shadow-тестом, blue-noise dither.
* Биолюминесценция: `emissive = palette(color_seed) * (0.5 + 0.5*sin(time*2 + phase))` -> HDR -> bloom.

### Небо
* Heightfield 1024x1024 `RG32Float` (height, biome_mask), генерится один раз compute-проходом:
  fbm simplex с domain warping, маска биома из worley-зон.
* Terrain рендерится vertex-pulling сеточным clipmap'ом; нормали — центральные разности карты.
* Дюны: ridged fbm + асимметричный профиль. Каньон: террасирование `h = floor(h*k)/k` со smooth blend.
* Деревья: compute-scatter по blue-noise с учётом biome_mask -> instance buffer; меш генерится
  один раз на CPU фрактально (3 уровня, ~200 tris).
* Атмосфера: аналитическое одиночное рассеяние Rayleigh + Mie (Henyey-Greenstein g=0.76).
* Палитры птиц: HSV-сдвиг по biome_mask под центроидом стаи.

### Переключение режимов
Все ресурсы обоих миров создаются на старте (десятки МБ VRAM). Swap = смена активного
`Box<dyn RenderMode>` + активного compute pipeline. Bind group layouts `integrate_fish` /
`integrate_bird` идентичны -> буферы и bind groups переиспользуются, device/surface не трогаем.
При переключении `SimParams.mode` меняется вместе с lerp-морфингом гравитации/границ за 1.5 с.

## 5. Математический аппарат

Полный вывод — в `docs/math.md`. Кратко:

```
Разделение:    F_sep = w_sep' * sum_j ( (p_i - p_j) / |p_i - p_j|^2 ),  |p_i - p_j| < r_sep
Выравнивание:  F_ali = w_ali' * ( normalize(mean(v_j)) * v_max - v_i )
Сплочённость:  F_coh = w_coh' * ( normalize(mean(p_j) - p_i) * v_max - v_i )
Адаптивные веса:
    w_sep' = w_sep * (1 + alpha * n_local / n_ref)      // плотно -> расталкивает сильнее
    w_coh' = w_coh * exp(-beta * n_local / n_ref)       // гасит коллапс ядра
    в панике (repel): w_ali' = 0.4 * w_ali, v_max' = 1.8 * v_max, tau ~ 1.5 s релаксация

SDF-огибание:
    grad SDF(p) ~ ( SDF(p+e_k) - SDF(p-e_k) ) / (2*eps),  eps = 0.5 * cell_size
    n = normalize(grad);  d = SDF(p)
    F_sdf = k_avoid * smoothstep(r_safe, 0, d) * n
    предиктивный зонд: p_ahead = p + v * t_look
        если SDF(p_ahead) < r_safe:  F_slide = k * normalize(v - n*dot(v,n))   // огибание, не торможение

Луч курсора:
    ndc = (2*mx/W - 1, 1 - 2*my/H)
    p_near = inv_view_proj * (ndc, 0, 1);  p_far = inv_view_proj * (ndc, 1, 1)   // с делением на w
    ro = cam_pos;  rd = normalize(p_far - p_near)
    плоскость: P0 = cam_pos + forward * d_focus,  n_p = -forward
    t = dot(P0 - ro, n_p) / dot(rd, n_p);  focus = ro + t*rd     (|dot(rd,n_p)| < 1e-6 -> промах)
    при экранировании рельефом: raymarch до пересечения -> фактическая точка фокуса

Воздействие курсора:
    g = focus - p_i;  r = |g|;  w(r) = clamp(1 - r/R, 0, 1)^2
    attract: F = +k_a * w(r) * normalize(g)
    repel:   F = -k_r * w(r)^1.5 * normalize(g) + k_t * w(r) * normalize(cross(g, up))

Интеграция (semi-implicit Euler + кламп длины):
    a = clamp_len(sum F, F_max);  v' = clamp_len(v + a*dt, v_min, v_max);  p' = p + v'*dt
```

## 6. Документация и стандарт кода

```
docs/
  README.md           сборка/запуск, управление, требования к GPU
  architecture.md     диаграмма крейтов и поток данных за кадр
  gpu-pipeline.md     таблица буферов, bind group layouts, порядок проходов, размеры dispatch
  math.md             формулы с выводом и единицами
  perf.md             timestamp-query замеры по проходам, бюджет кадра
  adr/0001..0005      Context / Decision / Consequences / Alternatives, <= 1 страницы
```

Правила комментирования:

* WGSL — у каждого файла шапка-контракт: имя прохода, все `@group/@binding` с типами,
  workgroup size, dispatch размер, инварианты (например «keys отсортированы по key»,
  «cell_start[c] == U32_MAX для пустой ячейки»), правила конкурентного доступа
  (что читается, что пишется, почему нет гонок). Формулы помечаются `// math.md §3.2`.
* Rust — над каждым POD таблица offset/size/WGSL-имя, ссылка на `docs/gpu-pipeline.md`
  и `const`-assert'ы вместо комментариев «поверь мне».
* `unsafe` — только derive `Pod/Zeroable`. Если ручной `unsafe impl` неизбежен, обязателен
  блок `// SAFETY:` с перечислением инвариантов. `#![deny(unsafe_op_in_unsafe_fn)]`.
* clippy::pedantic с точечными allow, rustfmt.

## 7. Пошаговый план на 4 дня

### День 1 — каркас и наивный рой
1. Workspace, 5 крейтов, pinned deps (wgpu, winit, glam, bytemuck).
2. `boids-core`: POD-структуры + const-assert'ы + CPU reference + камера/луч.
3. `boids-gpu`: init device/adapter, ping-pong буферы, compute `integrate_naive` (wander + границы).
4. `boids-render`: instanced draw low-poly меша, TBN из скорости в VS, Lambert + fog.
5. `boids-app`: winit `ApplicationHandler`, surface, resize, orbit-камера, FPS.

DoD: окно открывается, 100k инстансов летают и ориентируются по скорости, 60+ FPS на NVIDIA;
heatless-тесты layout зелёные; clippy чистый; `docs/architecture.md` + ADR-0001.

### День 2 — сетка, сортировка, настоящая стая
1. `hash` + `clear_cells` + `build_ranges`.
2. Bitonic sort: сначала глобальный, затем `sort_local` оптимизация.
3. Тесты: GPU readback vs CPU sort (10 сидов, включая N не кратное pow2); инварианты диапазонов.
4. `integrate` с обходом 27 ячеек, адаптивные веса; сверка с CPU-эталоном (N=1024, 20 шагов).
5. Timestamp queries -> `docs/perf.md`.

DoD: 100k формируют устойчивые стаи, sort-тесты зелёные, ADR-0002/0003.

### День 3 — курсор, SDF, океан
1. `ray_from_ndc` + `ray_plane_intersect` + unit-тесты.
2. `InteractionUniforms`, режимы attract/repel, визуализация точки фокуса.
3. SDF океана в WGSL + градиентное отталкивание + предиктивный зонд.
4. `FishMode`: raymarch окружения с записью глубины, Beer-Lambert, каустика, god rays, биолюм.
5. HDR + bloom + ACES tonemap.

DoD: рыбы огибают колонны без залипания, курсор управляет стаей, 100k >= 60 FPS, ADR-0005.

### День 4 — птицы, биомы, полировка
1. Compute heightfield + biome mask; terrain vertex-pulling + clipmap LOD.
2. Три биома переключением параметров шума; scatter деревьев.
3. Атмосферное рассеяние, палитры птиц, коллизии с рельефом через градиент карты высот.
4. Chromatic aberration, vignette, film grain в композите.
5. Морфинг Fish <-> Birds по TAB без пересоздания device/surface.
6. Финализация docs, скриншоты.

DoD: TAB мгновенно переключает мир, 3 наземных биома работают, все тесты зелёные, README
воспроизводим с нуля.

## 8. Валидация

* Компиляция шейдеров: `naga`-валидация всех `.wgsl` в тесте -> падает CI, а не рантайм.
* Layout: const-assert'ы + тест «Rust offset == WGSL offset» через пробный compute-проход.
* Сортировка: GPU readback vs CPU sort, включая N не кратное pow2.
* Сетка: монотонность диапазонов, покрытие [0, N), пустые ячейки = U32_MAX.
* Физика: GPU vs CPU-эталон на N=1024 (тот же сид): центроид, средняя скорость, order parameter.
* SDF: Rust-версия vs WGSL на сетке 32^3 точек, |Δ| < 1e-4.
* Перф: timestamp queries на проход, бюджет 16.6 мс, регрессионный лог в `docs/perf.md`.
* Локально (WSL, llvmpipe) — headless-тесты при N=1024..4096; визуал и 100k — на NVIDIA-машине.

## 9. Открытые вопросы

1. Bitonic vs radix: выбран bitonic (проще, детерминирован). Radix 4-bit — оптимизация Day 5,
   зафиксировано в ADR-0002 как отложенное.
2. Прямая индексация сетки вместо хеша (ADR-0003): плата 8.4 МБ на таблицы диапазонов.
3. Окружение океана: raymarch SDF принят; при упоре в fps — half-res + upsample.
4. egui-оверлей для тюнинга весов: включаем, если День 2 идёт по графику.
5. Детерминированный режим `--deterministic` (фиксированный сид и dt) для скриншот-регрессий.
6. Разрешение: адаптивное 1080p..4K, основной таргет 1440p. Half-res проходы для raymarch
   окружения и bloom включаются автоматически при > 1440p.
