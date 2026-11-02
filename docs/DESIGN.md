# Design system — "Porcelain / Graphite"

The interface is a precision instrument for an organization's knowledge: calm,
dense where it matters, editorial at its hero moments. Everything below is
implemented in `frontend/src/app/globals.css`, `frontend/tailwind.config.ts`
and `frontend/src/components/ui/*`. Build pages from those pieces; don't
restyle them locally.

## Principles

1. **One accent, used with intent.** Iris (`accent`) marks the primary action,
   the current location, AI activity and links. Status colours (`ok`, `warn`,
   `danger`, `info`) only ever describe state. Never decorate with them.
2. **Hairlines over boxes.** Prefer one card with hairline dividers to a grid of
   separate cards (`StatGroup`, `divide-y`, `divide-x`). Borders are 1px, warm
   in light and neutral in dark.
3. **Type does the hierarchy.** Size, weight and tracking create order before
   colour does. Mono micro-labels (`.label`) read like instrument readouts.
4. **Real content, real states.** Every list has loading (skeleton), empty
   (`EmptyState`, with a next step) and error states. Never show fake data, and
   never show a control that does nothing.
5. **Quiet motion.** 150–200 ms ease-out for UI, a 420 ms fade-up on page entry
   (`animate-fade-up`, `.stagger`). Respect `prefers-reduced-motion` (global).

## Tokens

| Token | Light (porcelain) | Dark (graphite) | Use |
|---|---|---|---|
| `frame` | warm grey | near-black | app frame + sidebar |
| `bg` | warm white | graphite | content panel background |
| `surface` | white | raised graphite | cards, inputs, menus |
| `surface-2` | porcelain | lighter graphite | fills, table headers, hover |
| `border` / `border-strong` | warm hairline | neutral hairline | dividers / hover borders |
| `fg` / `muted` / `subtle` | ink / 37% / 55% | 94% / 66% / 47% | text tiers |
| `accent` / `accent-fg` | iris 244° 74% 58% | iris 244° 95% 73% | primary + AI |
| `ok` `warn` `danger` `info` | — | — | state only |

All colours are `hsl(var(--x) / <alpha>)`, so opacity modifiers work: `bg-accent/10`,
`ring-ok/20`. An organization's branding accent overrides `--accent` at runtime
(`useOrgBranding`), with lightness clamped per theme. Never hard-code hex
colours in components; if a visualization needs a categorical palette, derive
it from tokens or define it once at the top of that module.

Shadows: `shadow-xs` (controls), `shadow-card` (cards, includes a 1px top
highlight), `shadow-lift` (hovered cards), `shadow-pop`/`shadow-float`
(menus, dialogs, popovers). Radii: `rounded-lg` (9px) controls, `rounded-xl`
(13px) cards, `rounded-2xl` (18px) dialogs/hero panels.

## Typography

- **Geist** (`font-sans`): all UI. Base 14px. Body copy 13.5–14.5px.
- **Geist Mono** (`font-mono`): ids, code, numbers in readouts, `.label`.
- **Instrument Serif** (`font-display`): hero moments only: the overview
  greeting, sign-in headline, large empty states. Never for UI controls,
  never below 28px.

| Role | Classes |
|---|---|
| Page title | `PageHeader` → 28px / 600 / `tracking-[-0.03em]` |
| Section title | `SectionTitle` → 15px / 600 |
| Card title | `CardHeader` → 14px / 600 |
| Micro label | `.label` → mono 10.5px uppercase, 0.12em tracking, `text-subtle` |
| Metric | `Stat` → 28px / 600 / tabular (`.num`) |
| Body | 13.5px `text-fg`; secondary 13px `text-muted`; meta 12px `text-subtle` |

Use `.num` (tabular numerals) for any column of numbers. Format durations as
human units (`1m 54s`, not `113742 ms`) and bytes with `formatBytes`.

## Layout

- App shell: sidebar on the `frame`; content on a raised `bg` panel with a
  48px breadcrumb bar (`Topbar`). Mobile: the panel goes full-bleed and the
  sidebar becomes a drawer.
- Every page is wrapped in `<Page>` (max 1320px, 32px gutters, responsive) and
  starts with `<PageHeader eyebrow title description actions>`. Use
  `width="narrow"` for reading/forms, `"wide"`/`"full"` for canvases.
- Vertical rhythm: 32px between page sections (`space-y-8`), 16–24px inside.
- Grids collapse to one column below `sm`; no horizontal page scroll at 375px
  (tables scroll inside their card).

## Components (from `components/ui`)

- `Button` variants: `primary` (one per surface), `secondary` (default for the
  rest), `ink` (hero/auth CTAs), `ghost` (toolbars), `danger`, `link`. Sizes
  `xs sm md lg icon`.
- `Card`, `CardHeader`, `CardBody`, `SectionTitle`.
- `StatGroup` + `Stat`: one card, hairline-divided cells. Standalone `Stat`
  still works for a single metric.
- `Badge` (pill, tinted, inset ring) and `StatusBadge` (status → tone + dot;
  `info` statuses pulse). `Dot` for inline live indicators.
- `Tabs`/`TabsList variant="line"` for page sections, default `segmented` for
  compact view switches.
- `DataTable`: searchable/sortable/paginated, mono uppercase headers, 48px rows,
  accent bar on the selected row.
- `EmptyState`: ring illustration, title, one-line explanation, next action.
- `Dialog`/`ConfirmDialog`, `Menu*`, `Tip` (ink tooltip), `Switch`, `Checkbox`,
  `Kbd`, `Avatar`, `Mono`, form `Input`/`Textarea`/`Select`/`Field`.

## Patterns

- **Page header actions**: at most one `primary`; others `secondary`.
- **Filters**: a toolbar row above content — search input first, then selects,
  then view toggles on the right.
- **Detail panels**: right-hand `Card` (`lg:sticky lg:top-6`) or a dialog; never
  navigate away for a quick look.
- **Live data**: `Dot live` + relative time (`timeAgo`), mono for timestamps.
- **Permissions**: hide navigation the user can't use; disable (not hide) an
  action on a visible screen, with a `Tip` explaining which permission is needed.
- **Copy**: calm, specific, sentence case. Say what happens and what to do next.
  No exclamation marks, no marketing superlatives.

## Accessibility

Contrast ≥ 4.5:1 for text (`subtle` is for meta text ≥ 12px only). Every icon
button has an `aria-label`; focus rings are always visible (`:focus-visible`
uses `--ring`). Interactive rows are buttons or links, not bare `div`s.
