// =========================================
// ID: PHYLOFACE_LIB_PDF_REPORT
// VERSION: v1.0
// =========================================
// Genera el informe de la App primaria (#31) como PDF, 100% CLIENT-SIDE (jsPDF):
// las imágenes nunca salen del browser, el PDF se arma en memoria y se descarga.
// Contenido: las 3 caras (alineadas) + veredicto global (coseno + posterior
// calibrado #6) + herencia por región (del método mostrado) + disclaimer.
//
// Las caras llegan como dataURL (PNG) ya rasterizadas por el caller desde la cara
// alineada 112×112 (ver AppPrimaria.alignedToDataUrl). El veredicto llega ya
// computado (lib/verdict) para no acoplar este módulo al motor.

import { jsPDF } from 'jspdf';
import type { GlobalVerdict, RegionalVerdict, Side } from './verdict';

export interface ReportFace { label: string; dataUrl: string; }

export interface ReportData {
  faces: { padre?: ReportFace; child: ReportFace; madre?: ReportFace };
  global: GlobalVerdict;
  regional: RegionalVerdict | null;
  /** Etiquetas de los lados ('Padre'/'Madre'). */
  parentLabels: Record<Side, string>;
}

const PADRE_RGB: [number, number, number] = [37, 99, 235];
const MADRE_RGB: [number, number, number] = [22, 163, 74];

function globalHeadline(g: GlobalVerdict, labels: Record<Side, string>): string {
  const present = (['left', 'right'] as Side[]).filter((s) => g.cosine[s] != null);
  if (present.length < 2) {
    const s = present[0];
    return s ? `Comparación con ${labels[s]}` : 'Sin datos';
  }
  if (g.winner === 'tie') return 'Se parece de forma pareja a ambos';
  return `Se parece más a ${labels[g.winner === 'left' ? 'left' : 'right']}`;
}

export function generateReportPdf(data: ReportData): void {
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });
  const W = 210, M = 15;
  const contentW = W - 2 * M;

  // --- Título + fecha ---
  doc.setFont('helvetica', 'bold'); doc.setFontSize(18); doc.setTextColor(20, 20, 20);
  doc.text('Informe de parecido facial', M, 20);
  doc.setFont('helvetica', 'normal'); doc.setFontSize(10); doc.setTextColor(120, 120, 120);
  doc.text(`App primaria · ${new Date().toLocaleString('es-AR')}`, M, 27);

  // --- Fila de caras: Padre · Hijo/a · Madre ---
  const faceSize = 42;
  const gap = (contentW - 3 * faceSize) / 2;
  const cells: { face?: ReportFace; side: Side | null }[] = [
    { face: data.faces.padre, side: 'left' },
    { face: data.faces.child, side: null },
    { face: data.faces.madre, side: 'right' },
  ];
  const faceY = 37;
  let fx = M;
  for (const cell of cells) {
    const cx = fx + faceSize / 2;
    if (cell.face) {
      doc.addImage(cell.face.dataUrl, 'PNG', fx, faceY, faceSize, faceSize);
      const rgb = cell.side === 'left' ? PADRE_RGB : cell.side === 'right' ? MADRE_RGB : [60, 60, 60] as [number, number, number];
      doc.setFont('helvetica', 'bold'); doc.setFontSize(11); doc.setTextColor(...rgb);
      doc.text(cell.face.label, cx, faceY + faceSize + 6, { align: 'center' });
      if (cell.side) {
        const cos = data.global.cosine[cell.side];
        const post = data.global.posterior[cell.side];
        doc.setFont('helvetica', 'normal'); doc.setFontSize(9); doc.setTextColor(90, 90, 90);
        const parts = [`cos ${cos != null ? cos.toFixed(3) : '—'}`];
        if (post != null) parts.push(`parentesco ${Math.round(post * 100)}%`);
        doc.text(parts.join('  ·  '), cx, faceY + faceSize + 11, { align: 'center' });
      }
    } else {
      doc.setDrawColor(210); doc.setLineWidth(0.3); doc.rect(fx, faceY, faceSize, faceSize);
      doc.setFontSize(9); doc.setTextColor(180, 180, 180);
      doc.text('—', cx, faceY + faceSize / 2, { align: 'center' });
    }
    fx += faceSize + gap;
  }

  let y = faceY + faceSize + 24;

  // --- Veredicto global ---
  const winnerRgb = (data.global.winner === 'left' && data.global.cosine.left != null && data.global.cosine.right != null)
    ? PADRE_RGB
    : (data.global.winner === 'right' && data.global.cosine.left != null && data.global.cosine.right != null)
      ? MADRE_RGB
      : [40, 40, 40] as [number, number, number];
  doc.setFont('helvetica', 'bold'); doc.setFontSize(16); doc.setTextColor(...winnerRgb);
  doc.text(globalHeadline(data.global, data.parentLabels), M, y);
  y += 6;
  doc.setFont('helvetica', 'normal'); doc.setFontSize(9); doc.setTextColor(120, 120, 120);
  doc.text('Parecido global por cara completa (coseno) + probabilidad calibrada de parentesco (KinFaceW-I).', M, y);
  y += 9;

  // --- Herencia por región ---
  doc.setFont('helvetica', 'bold'); doc.setFontSize(13); doc.setTextColor(30, 30, 30);
  doc.text('¿Por qué? — herencia por región', M, y);
  y += 7;

  const row = (label: string, rgb: [number, number, number], groups: string[]) => {
    doc.setFont('helvetica', 'bold'); doc.setFontSize(10); doc.setTextColor(...rgb);
    doc.text(label, M, y);
    doc.setFont('helvetica', 'normal'); doc.setTextColor(60, 60, 60);
    const text = groups.length ? groups.join(', ') : '—';
    const lines = doc.splitTextToSize(text, contentW - 42);
    doc.text(lines, M + 42, y);
    y += Math.max(6, lines.length * 5);
  };

  if (data.regional) {
    row(`Heredó de ${data.parentLabels.left}:`, PADRE_RGB, data.regional.inheritedLeft.map((g) => g.label));
    row(`Heredó de ${data.parentLabels.right}:`, MADRE_RGB, data.regional.inheritedRight.map((g) => g.label));
    if (data.regional.balanced.length) row('Equilibrado:', [120, 120, 120], data.regional.balanced.map((g) => g.label));
    doc.setFont('helvetica', 'italic'); doc.setFontSize(8); doc.setTextColor(140, 140, 140);
    doc.text(`Según ${data.regional.methodLabel} (confiabilidad ${data.regional.confidence}).`, M, y + 1);
  } else {
    doc.setFont('helvetica', 'italic'); doc.setFontSize(9); doc.setTextColor(140, 140, 140);
    doc.text('Sin desglose por región calculado.', M, y);
  }

  // --- Footer ---
  doc.setFont('helvetica', 'normal'); doc.setFontSize(8); doc.setTextColor(150, 150, 150);
  doc.text(
    'Generado localmente en tu navegador — las imágenes no se subieron a ningún servidor.',
    M, 287,
  );

  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
  doc.save(`informe-parecido-${stamp}.pdf`);
}
