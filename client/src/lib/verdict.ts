// =========================================
// ID: PHYLOFACE_LIB_VERDICT
// VERSION: v1.0
// =========================================
// Síntesis interpretable de la App primaria (Tarea #12): combina la comparación
// GLOBAL (coseno del Hijo/a contra cada progenitor, + posterior calibrado de la
// Tarea #6) y la comparación REGIONAL (reparto P↔M por grupo facial) en una
// lectura directa que responde la pregunta del producto: "¿a quién se parece y
// por qué?".
//
// Es PURO (sin React, sin DOM): toma resultados ya calculados y devuelve un
// objeto `Verdict` que la UI renderiza. El veredicto regional se deriva con los
// MISMOS helpers que dibujan las barras/radar (lib/regionalAggregate) para que
// nunca diverja de lo que el usuario ve abajo.
//
// Lados: 'left'/'right' son los slots del progenitor (la App primaria los mapea
// a "Padre"/"Madre" según el layout); este módulo no asume sexos.
//
// Honestidad: el veredicto regional viaja con el `method` que lo produjo
// (geométrico = forma; occlusion = contribución) y su `confidence`; la UI debe
// mostrarlo. No mezcla métodos: el resumen refleja el método actualmente activo.

import {
  DISPLAY_GROUPS, GROUP_LABEL, rawScoreMap, perRegionValues, groupValues,
} from './regionalAggregate';
import type {
  RegionalScoresResult, RegionalMethod, ScoreConfidence,
} from './regionalScores';
import type { ValueScore } from './calibration';

export type Side = 'left' | 'right';
export type Winner = 'left' | 'right' | 'tie';

// -----------------------------------------
// Veredicto GLOBAL: cara completa.
// -----------------------------------------
export interface GlobalVerdict {
  cosine: Partial<Record<Side, number>>;
  /** Posterior calibrado P(parentesco|coseno) por lado (Tarea #6), si hay
   *  calibración cargada. Relación 'ALL' (no se conoce el sexo del Hijo/a). */
  posterior: Partial<Record<Side, number>>;
  /** Veredicto duro vs umbral de Youden por lado, si hay calibración. */
  isKin: Partial<Record<Side, boolean>>;
  winner: Winner;
  /** Diferencia absoluta de coseno entre lados (0 si falta uno). */
  marginCosine: number;
}

// -----------------------------------------
// Veredicto por GRUPO facial (8 grupos colapsados).
// -----------------------------------------
export interface GroupVerdict {
  key: string;
  label: string;
  /** Reparto 0..1 por lado (escala relativa: suman ~1 entre left y right). */
  share: Partial<Record<Side, number>>;
  winner: Winner;
  /** |share_left − share_right|: cuán marcada es la herencia hacia el ganador. */
  margin: number;
}

export interface RegionalVerdict {
  method: RegionalMethod;
  methodLabel: string;
  confidence: ScoreConfidence;
  groups: GroupVerdict[];          // todos los grupos válidos, orden anatómico
  inheritedLeft: GroupVerdict[];   // grupos que gana 'left', orden margen desc
  inheritedRight: GroupVerdict[];  // grupos que gana 'right', orden margen desc
  balanced: GroupVerdict[];        // empates (margen < tieEps)
}

export interface Verdict {
  global: GlobalVerdict;
  regional: RegionalVerdict | null; // null hasta que haya scores regionales
}

// Umbral de empate global por coseno: por debajo, "se parece parecido a ambos".
const GLOBAL_TIE_EPS = 0.02;
// Umbral de empate regional por reparto (|L−R| de shares que suman ~1).
const REGIONAL_TIE_EPS = 0.06;

function pickWinner(l: number | undefined, r: number | undefined, eps: number): Winner {
  const hasL = l != null && !Number.isNaN(l);
  const hasR = r != null && !Number.isNaN(r);
  if (hasL && hasR) {
    if (Math.abs((l as number) - (r as number)) < eps) return 'tie';
    return (l as number) > (r as number) ? 'left' : 'right';
  }
  if (hasL) return 'left';
  if (hasR) return 'right';
  return 'tie';
}

// -----------------------------------------
// Veredicto global desde los cosenos y (opcional) los scores calibrados por lado.
// -----------------------------------------
export function buildGlobalVerdict(
  cosineLeft: number | undefined,
  cosineRight: number | undefined,
  scoreLeft?: ValueScore,
  scoreRight?: ValueScore,
): GlobalVerdict {
  const cosine: Partial<Record<Side, number>> = {};
  const posterior: Partial<Record<Side, number>> = {};
  const isKin: Partial<Record<Side, boolean>> = {};
  if (cosineLeft != null) cosine.left = cosineLeft;
  if (cosineRight != null) cosine.right = cosineRight;
  if (scoreLeft) { posterior.left = scoreLeft.posterior; isKin.left = scoreLeft.isKin; }
  if (scoreRight) { posterior.right = scoreRight.posterior; isKin.right = scoreRight.isKin; }
  const margin =
    cosineLeft != null && cosineRight != null ? Math.abs(cosineLeft - cosineRight) : 0;
  return {
    cosine, posterior, isKin,
    winner: pickWinner(cosineLeft, cosineRight, GLOBAL_TIE_EPS),
    marginCosine: margin,
  };
}

// -----------------------------------------
// Veredicto regional desde los resultados por lado del scorer activo. Usa el
// reparto P↔M (escala relativa) y los grupos de display compartidos con el panel.
// -----------------------------------------
export function buildRegionalVerdict(
  bySide: Partial<Record<Side, RegionalScoresResult>>,
  meta: { method: RegionalMethod; methodLabel: string; confidence: ScoreConfidence },
): RegionalVerdict {
  const rawL = rawScoreMap(bySide.left);
  const rawR = rawScoreMap(bySide.right);
  const per = perRegionValues(rawL, rawR, 'relative');
  const leftG = groupValues(per.left);
  const rightG = groupValues(per.right);

  const groups: GroupVerdict[] = [];
  for (const g of DISPLAY_GROUPS) {
    const l = leftG.get(g.key)?.value;
    const r = rightG.get(g.key)?.value;
    const hasL = l != null && !Number.isNaN(l);
    const hasR = r != null && !Number.isNaN(r);
    if (!hasL && !hasR) continue; // grupo sin datos válidos (ambas regiones inválidas)
    const share: Partial<Record<Side, number>> = {};
    if (hasL) share.left = l as number;
    if (hasR) share.right = r as number;
    const margin = hasL && hasR ? Math.abs((l as number) - (r as number)) : 1;
    groups.push({
      key: g.key,
      label: GROUP_LABEL[g.key] ?? g.key,
      share,
      winner: pickWinner(l, r, REGIONAL_TIE_EPS),
      margin,
    });
  }

  const byMarginDesc = (a: GroupVerdict, b: GroupVerdict) => b.margin - a.margin;
  return {
    ...meta,
    groups,
    inheritedLeft: groups.filter((g) => g.winner === 'left').sort(byMarginDesc),
    inheritedRight: groups.filter((g) => g.winner === 'right').sort(byMarginDesc),
    balanced: groups.filter((g) => g.winner === 'tie').sort(byMarginDesc),
  };
}
