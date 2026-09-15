# GPU Boids 3D (50k–100k агентов) на чистом Rust + wgpu

Технический документ и план реализации. Спринт: 4 дня плотной разработки.
Статус: **Day 1 — done**, **Day 2 — done**, **Day 3 — done**, Day 4 — next.

## Статус на конец Дня 2

Сделано в День 2 (всё проверено тестами, `cargo test --workspace`):
* spatial grid на direct-index: `clear_cells`, `hash`, `build_ranges`, инварианты диапазонов
  проверены device-тестом на покрытие и пустые ячейки;
* bitonic sort: 153 стадии для 131072 ключей, по одному compute-pass на стадию, параметры стадии
  через `var<immediate>` + `set_immediates` (один pipeline и bind group на все стадии; отдельный
  device-тест доказывает, что immediates доезжают до шейдера per-dispatch);
* `KeyPlan` прибивает отсортированный массив к `keys[0]` независимо от чётности числа стадий;
* GPU-sort сверяется с CPU-битоником на N, не кратных pow2, и на обеих чётностях стадий;
* `integrate_grid` обходит 27 ячеек и сходится с all-pairs поштучно за 1 шаг и по агрегатам за 40;
* unprepared grid находит ноль соседей (не читает мусор) — отдельный тест;
* `Strategy::for_count` + `--strategy naive|grid`, порог `NAIVE_AGENT_LIMIT = 1024` замерен
  `--bench` (кросовер чуть ниже 1536 агентов), дефолт приложения — 100000 агентов;
* `--bench <N>`: headless-прогон с per-pass GPU-таймингами через timestamp queries, из него
  пересобираются числа в `docs/perf.md`.

Сделано в День 1:
* workspace из 5 крейтов, `boids-core` без зависимости от `wgpu`;
* контракт раскладки GPU-структур с `const`-assert'ами и **device-тестом**, который сверяет
  смещения полей в WGSL с Rust `offset_of!`;
* CPU-эталон шага симуляции; GPU сходится с ним поштучно за 1 шаг и по агрегатам за 600 шагов;
* симуляция целиком на GPU: ping-pong буферы, один compute-проход на кадр, CPU не перебирает агентов;
* instanced-рендер: меш и базис ориентации считаются в vertex shader, ни vertex buffer, ни
  instance buffer нет;
* оба мира: процедурный фон, параметры меша и среды под режим, переключение по TAB без переаллокаций;
* интерактив курсором: аттрактор, репеллер с закруткой, временный override на средней кнопке;
* headless-скриншоты в PNG (`--screenshot`), тесты на пиксели и глубину кадра;
* документация: README, architecture, gpu-pipeline, math, perf, ADR-0001/0002.

Не сделано (план на Дни 3-4 без изменений):
* рендер поверхности и коллизии с окружением (силы и поля написаны и протестированы, но
  `SimConfig::env` ещё не связан с рендером этих поверхностей);
* объёмный подводный рендер, god rays, каустика, bloom;
* рельеф с биомами, атмосферное рассеяние, морфинг переключения миров.

Замеры и их ограничения: `docs/perf.md`. Важное: локально доступны только `llvmpipe` (Vulkan) и
GL-адаптер, который через Mesa `d3d12` попадает на встроенную Radeon и не презентит, поэтому 100k
агентов и реальный FPS надо мерить на целевой машине. Числа `--bench` — это встроенная графика
через D3D12→GL, не целевая 4060 Ti.

## Статус на конец Дня 3

Сделано в День 3 (всё проверено `cargo test --workspace`):
* дефекты §10 закрыты: `spawn_swarm` собирает одну плотную стаю эллипсоидом в центре мира, а
  rejection sampling держит минимальную дистанцию `r_sep`; спавн по-прежнему детерминирован по сиду
  и общий для CPU и GPU;
* курсор: `ray_from_ndc` + `ray_plane_intersect`, режимы attract/repel, точка фокуса рисуется в
  окружении и перекрывается тем, во что попал raymarch (`render/focus_marker_is_visible`);
* SDF-огибание: градиент + предиктивный зонд, и поля CPU/WGSL сверены device-тестом
  `sdf/wgsl_matches_rust` на сетке 32³ (|Δ| < 1e-4); заодно CPU-двойник `reef_field` приведён
  к WGSL (вторая семья колонн, радиусы `smin` 3 и 6);
* `FishMode`: full-screen sphere trace по тому же `reef_field`, что и симуляция, пишет `frag_depth`,
  поэтому рыбы перекрываются рифом; Beer-Lambert по каналам, каустика, god rays, биолюминесценция
  (`render/ocean_writes_depth`, `render/underwater_is_blue_and_lit`);
* HDR: геометрические проходы пишут `Rgba16Float`; bloom-пирамида из трёх уровней с аддитивным
  upsample; композит с ACES, хроматической аберрацией, виньеткой и зерном
  (`render/bloom_adds_light`, ADR-0004);
* оба мира построены на старте, `TAB` меняет активный pipeline без пересоздания device/surface;
* render suite: 8 проверок, включая глубину, голубизну воды, вклад bloom и маркер фокуса.

Не сделано (День 4):
* рельеф с биомами: compute heightfield + biome mask, vertex-pulling clipmap, scatter деревьев,
  атмосферное рассеяние;
* морфинг Fish <-> Birds за 1.5 с (сейчас переключение мгновенное);
* коллизии птиц с нарисованным рельефом: поле есть, меша ещё нет.

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
  `prev_dir: [f32;3]`, `color_seed: f32` (prev_dir — для bank-угла в VS).
* `KeyVal { key: u32, val: u32 }` — 8 B, ключ сортировки = linear cell index.
* `SimParams` (144 B), `InteractionUniforms` (48 B), `CameraUniform`, `SortParams` — все align(16).
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
    read: usize,                   // индекс актуального буфера
}
pub struct SimPipelines { clear_cells, hash, sort, build_ranges,
                          integrate_naive, integrate_grid }
pub struct KeyPlan { hash_dst: usize, stages: u32 }  // прибивает результат сортировки к keys[0]
pub fn record_step(enc, res, strategy, profiler);    // grid prep + integrate, либо только integrate
pub struct ShaderCache;              // include-препроцессор + опциональный hot-reload
```

Параметры стадии сортировки идут не через uniform с dynamic offset, а через `var<immediate>` +
`set_immediates`; `sort_local` (tile в shared memory) не понадобился и отложен вместе с radix
(ADR-0003).

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
| `SimParams` | 144 | `struct SimParams` | каждый vec3 + f32/u32 |
| `InteractionUniforms` | 48 | `struct Interaction` | ray_origin+f32, focus+f32, ... |
| `SortParams` | 16 | `struct SortParams` | j, k, n, pad |

Тест `layout_matches_wgsl` пишет `offset_of`-значения из WGSL compute-прохода в буфер и сравнивает
с Rust — расхождение ловится в CI, а не в рантайме.

### 3.2 Буферы

| Буфер | Размер @100k | usage |
|---|---|---|
| `boids[2]` | 2 x 6.0 MB (48 B x 131072) | STORAGE, COPY_SRC, COPY_DST |
| `keys[2]` | 2 x 1.0 MB (8 B x 131072) | STORAGE, COPY_SRC |
| `cell_start`, `cell_end` | 2 x 1.3 MB (91x40x91 u32, дефолтный мир) | STORAGE, COPY_SRC |
| `params`, `interaction` | < 1 KB | UNIFORM, write_buffer раз в кадр |

Домен ограничен `bounds_half`, сетка выводится из него и `r_percept` (`GridDims::for_domain`,
не больше 256 ячеек на ось) -> прямая индексация без хеша (нет коллизий). Размер ячейки равен
`r_percept` — это требование корректности 27-ячеечного поиска, а не тюнинг. Пустая ячейка = `U32_MAX`
в `cell_start`.

### 3.3 Порядок compute-проходов кадра

```
P0 clear_cells   dispatch ceil(num_cells/256)      cell_start[c] = U32_MAX
P1 hash          dispatch ceil(n_padded/256)        keys[dst][i] = KeyVal{cell_index(pos), i}; padding -> U32_MAX
P2 sort          по dispatch на стадию (k,j)        параметры через immediates, один pipeline
P3 build_ranges  dispatch ceil((n_padded+1)/256)    диапазоны ячеек + терминатор последнего run
P4 integrate     один поток на boid: 3x3x3 ячеек -> силы -> SDF -> курсор -> интеграция
```

Sort: N padded до 131072 -> 17*18/2 = 153 стадии, каждая отдельным compute-pass (барьер между
проходами и есть синхронизация). `KeyPlan` прибивает результат к `keys[0]` независимо от чётности
числа стадий. `sort_local` (tile в shared memory) не понадобился; при упоре в 5.7 мс — radix
(ADR-0003).

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

1. Bitonic vs radix: выбран bitonic (проще, детерминирован; 153 стадии = 5.7 мс при 100k на
   локальном адаптере). Radix — оптимизация Дня 5, зафиксировано в ADR-0003 как отложенное.
2. Прямая индексация сетки вместо хеша (ADR-0003): плата 2.6 МБ на обе таблицы диапазонов при
   дефолтном мире (91×40×91 ячейка), кросовер naive/grid — чуть ниже 1536 агентов.
3. Окружение океана: raymarch SDF принят; при упоре в fps — half-res + upsample.
4. egui-оверлей для тюнинга весов: включаем, если День 2 идёт по графику.
5. Детерминированный режим `--deterministic` (фиксированный сид и dt) для скриншот-регрессий.
6. Разрешение: адаптивное 1080p..4K, основной таргет 1440p. Half-res проходы для raymarch
   окружения и bloom включаются автоматически при > 1440p.

## 10. Дефекты, требующие исправления

Взять в День 3 до SDF-работы: от этого зависит, что вообще видно на экране.
Статус: **исправлено** (День 3) — один плотный рой, минимальная дистанция при спавне, тесты
`spawn/*` и скриншот-прогоны зелёные.

1. **Стая на старте — не стая.** `spawn::spawn_swarm` (`boids-core/src/spawn.rs`) раскладывает
   агентов равномерно по коробке 60% от `bounds_half`. При такой плотности в `r_percept` у каждого
   агента оказывается всего ~5-10 соседей, и рой на первых секундах распадается на множество
   микро-стай вместо одной большой. Надо: на старте собирать всех в одну компактную плотную стаю
   (шар/эллипсоид) в центре мира, а не заполнять куб. Радиус стаи подбирать так, чтобы внутри
   `r_percept` было достаточно соседей (ориентир — `density_ref` и `SimConfig::dense`) и чтобы
   стая не выходила за soft-bounds на первых шагах.
2. **Агенты не должны пересекаться при спавне.** Сейчас `rng.in_box` может положить двух агентов
   почти в одну точку, и первый шаг уходит на их расталкивание. Нужна минимальная дистанция между
   позициями: rejection sampling по `r_sep` (или решётка с джиттером). Спавн остаётся
   детерминированным по сиду и по-прежнему общим для CPU и GPU — иначе поштучная сверка с эталоном
   теряет смысл.
3. **Не сломать существующее.** `spawn_swarm` используют app, screenshot и обе стороны CPU/GPU-сверки;
   `SimConfig::dense` — тесты. После смены раскладки прогнать `spawn_is_deterministic_and_inside_the_world`,
   `reference/*`, `grid/*` и скриншоты, и добавить два теста: «одна стая» (у большинства агентов на
   первом шаге есть хотя бы один сосед в `r_percept`) и «нет пересечений» (попарные дистанции не
   меньше порога).
