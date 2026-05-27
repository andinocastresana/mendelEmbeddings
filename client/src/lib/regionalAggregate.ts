// =========================================
// ID: PHYLOFACE_LIB_REGIONAL_AGGREGATE
// VERSION: v1.0
// =========================================
// Helpers PUROS de agregación de scores regionales: colapso de las 12 regiones
// canónicas en 8 grupos de display (pares L/R fusionados), reparto P↔M por
// región/grupo, y agregación a media + rango por grupo.
//
// Por qué existe (extraído de RegionalScoresPanel.tsx, 2026-05-27, Tarea #12):
// estas funciones eran privadas del panel. Al construir el veredicto de la App
// primaria (lib/verdict.ts) necesitábamos derivar "quién hereda cada región"
// con EXACTAMENTE la misma lógica que dibujan las barras/radar — si el panel y
// el veredicto divergieran, el resumen mentiría sobre lo que el usuario ve.
// Moverlas a un módulo compartido garantiza una sola fuente de verdad. Son
// puras (sin React, sin DOM) → testeables y reusables por panel + veredicto.
//
// Semántica del "Reparto P↔M" (escala relativa): por región,
//   share_L = score_L / (score_L + score_R)   (suma 1 entre los dos lados)
// responde "¿esta región se parece más al progenitor izquierdo o al derecho?".
// La escala "absoluta" devuelve el score crudo 0..1 de cada método sin repartir.

import { CANONICAL_REGIONS, type RegionName } from './regions';
import type { RegionalScoresResult } from './regionalScores';

export type Scale = 'relative' | 'absolute';

// -----------------------------------------
// Grupos de display: colapsan los pares L/R en una sola fila/eje. Se derivan del
// campo `group` del contrato canónico, preservando el orden anatómico. Las
// regiones de la línea media (nariz, boca, mentón, frente) quedan como grupos de 1.
// -----------------------------------------
export interface DisplayGroup { key: string; labelEs: string; regions: RegionName[]; }

export const GROUP_LABEL: Record<string, string> = {
  eyebrow: 'cejas', eye: 'ojos', cheekbone: 'pómulos', cheek: 'mejillas',
  nose: 'nariz', mouth: 'boca', chin: 'mentón', forehead: 'frente',
};

// Orden de display arriba→abajo por posición anatómica (pedido del usuario):
// frente · cejas · ojos · nariz · pómulos · mejillas · boca · mentón.
export const GROUP_ORDER = ['forehead', 'eyebrow', 'eye', 'nose', 'cheekbone', 'cheek', 'mouth', 'chin'];

export const DISPLAY_GROUPS: DisplayGroup[] = (() => {
  const byGroup = new Map<string, RegionName[]>();
  for (const r of CANONICAL_REGIONS) {
    if (!byGroup.has(r.group)) byGroup.set(r.group, []);
    byGroup.get(r.group)!.push(r.name);
  }
  // Grupos no listados en GROUP_ORDER (no debería haber) van al final, por las dudas.
  const leftovers = [...byGroup.keys()].filter((g) => !GROUP_ORDER.includes(g));
  return [...GROUP_ORDER.filter((g) => byGroup.has(g)), ...leftovers]
    .map((g) => ({ key: g, labelEs: GROUP_LABEL[g] ?? g, regions: byGroup.get(g)! }));
})();

// Valor agregado de un grupo: media de sus regiones válidas + rango [min,max]
// (la asimetría L/R del par). count<=1 → sin rango.
export interface GroupVal { value: number; min: number; max: number; count: number; }

// -----------------------------------------
// raw → score crudo 0..1 por región (NaN si inválida).
// -----------------------------------------
export function rawScoreMap(result: RegionalScoresResult | undefined): Map<RegionName, number> {
  const m = new Map<RegionName, number>();
  if (!result) return m;
  for (const s of result.scores) m.set(s.region, s.valid ? s.score : NaN);
  return m;
}

// Reparto por región: share = score / (scoreL + scoreR), suma 1 entre L y R.
// Con un solo progenitor presente, su share es 1 (100%). Ambos 0 → empate 0.5.
export function perRegionValues(
  rawL: Map<RegionName, number>,
  rawR: Map<RegionName, number>,
  scale: Scale,
): { left: Map<RegionName, number>; right: Map<RegionName, number> } {
  if (scale === 'absolute') return { left: rawL, right: rawR };
  const left = new Map<RegionName, number>();
  const right = new Map<RegionName, number>();
  for (const r of CANONICAL_REGIONS) {
    const l = rawL.get(r.name);
    const rr = rawR.get(r.name);
    const hasL = l != null && !Number.isNaN(l);
    const hasR = rr != null && !Number.isNaN(rr);
    if (hasL && hasR) {
      const sum = (l as number) + (rr as number);
      if (sum > 1e-9) { left.set(r.name, (l as number) / sum); right.set(r.name, (rr as number) / sum); }
      else { left.set(r.name, 0.5); right.set(r.name, 0.5); }
    } else if (hasL) { left.set(r.name, 1); }
    else if (hasR) { right.set(r.name, 1); }
  }
  return { left, right };
}

export function groupValues(perRegion: Map<RegionName, number>): Map<string, GroupVal> {
  const out = new Map<string, GroupVal>();
  for (const g of DISPLAY_GROUPS) {
    const vals = g.regions
      .map((r) => perRegion.get(r))
      .filter((v): v is number => v != null && !Number.isNaN(v));
    if (vals.length === 0) { out.set(g.key, { value: NaN, min: NaN, max: NaN, count: 0 }); continue; }
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    out.set(g.key, { value: mean, min: Math.min(...vals), max: Math.max(...vals), count: vals.length });
  }
  return out;
}
