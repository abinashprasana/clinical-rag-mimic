# Evidence Aperture identity

Evidence Aperture represents clinical records opening around one focused, reviewable evidence window. The mark identifies evidence inspection; it never communicates diagnosis, approval, certainty, or clinical status.

## Master artwork

- Use `evidence-aperture-color.svg` as the primary full-color, container-free mark on deep ink and navy surfaces.
- Use `evidence-aperture-mono-light.svg` when a single light ink is required on a dark surface.
- Use `evidence-aperture-mono-dark.svg` on light surfaces.
- `evidence-aperture-on-dark.svg` and `evidence-aperture-on-light.svg` are presentation proofs of the approved background pairings, not container variants of the primary mark.
- Use `evidence-aperture-scale-sheet.svg` to review the supplied sizes.
- Use `../favicon.svg` only in browser and launcher contexts. Its compact deep-ink canvas protects the canonical four-part mark on both light and dark browser chrome at 16px.

The wordmark remains live IBM Plex Sans text. Beside “Clinical Evidence Assistant,” the SVG is decorative and must retain `aria-hidden="true"`; the enclosing home link supplies the accessible name.

## Construction and spacing

- The master occupies a 32 × 32 grid with at least 2px of outer safe area and consistent 2px strokes.
- Its grid discipline follows [IBM UI icon design guidance](https://www.ibm.com/design/language/iconography/ui-icons/design/) while retaining original Evidence Aperture geometry.
- The grey rear record and cyan review record are asymmetric open frames with softly rounded exterior corners and square stroke caps.
- The off-white evidence window is a compact rounded outline. Its single cyan passage is the only internal detail.
- All four elements are stroke-only; do not fill the frames or evidence window.
- Preserve clear space equal to 8px at the 32px master size on every side, scaling that space proportionally.
- Keep the complete two-frame silhouette optically centered. Do not align to either record frame in isolation.
- Do not redraw, crop, rotate, skew, stretch, or alter the frame offsets, window, or passage proportions.

## Minimum sizes

- Primary full-color or monochrome mark: 20px minimum.
- Favicon-specific artwork: 16px minimum.
- The approved scale proof covers 16, 20, 24, 32, 64, and 128px. At 16px, use only the favicon artwork.

## Color

| Role | Value | Use |
| --- | --- | --- |
| Review cyan | `#62C7D0` | Review frame and focused passage |
| Record grey | `#B7C8CE` | Rear record frame |
| Off-white | `#F4F7F8` | Evidence-window outline, light monochrome, and light proof surface |
| Deep ink | `#071119` | Preferred dark surface and dark monochrome |

The full-color mark is intended for dark surfaces. Use the supplied dark monochrome artwork on light surfaces so the complete silhouette remains legible.

## Motion

- Product motion may reveal the rear frame, review frame, and evidence window in sequence once on initial load.
- The complete reveal must settle within 260ms and must not replay during navigation.
- Do not loop, pulse, shimmer, rotate, morph, or continuously animate any part of the mark.
- Under reduced motion, render the final static mark immediately.

## Prohibited alterations

- Do not enclose the primary mark in a tile, badge, shield, lozenge, or decorative container.
- Do not reinterpret the evidence window as a medical cross, checkmark, approval seal, status indicator, or form checkbox.
- Do not add hearts, brains, sparkles, chat bubbles, waveforms, node clusters, or decorative particles.
- Do not add fills, gradients, glow, shadows, bevels, textures, photography, or extra passage lines.
- Do not recolor the mark with clinical status green, amber, or red.

## Originality note

The offset-record and focused-window construction was created for the Clinical Evidence Assistant identity. A preliminary visual comparison considered the current identities published by [Abridge](https://www.abridge.com/), [Qdrant](https://qdrant.tech/brand-resources/), and [Weaviate](https://weaviate.io/company/playbook/creating-a-playful-new-identity-for-weaviate). No obvious close silhouette was identified. Preserve the supplied proportions because layered documents and focus frames remain common visual concepts. This screening is not exhaustive and is not legal trademark clearance.
