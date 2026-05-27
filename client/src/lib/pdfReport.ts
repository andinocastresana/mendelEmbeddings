// =========================================
// ID: PHYLOFACE_LIB_PDF_REPORT
// VERSION: v2.0
// =========================================
// Genera el informe de la App primaria (#31) como PDF, 100% CLIENT-SIDE.
//
// v2.0 (pedido del usuario): en lugar de re-dibujar un layout A4 a mano, RASTERIZA
// EL DOM REAL del informe con html2canvas y lo pagina en A4 con jsPDF, para que el
// PDF "respete lo que se ve en la página" (mismas caras, barras, radar, heatmap,
// herencia por región). Lo NO esencial para un documento estático (botones, inputs,
// solapas, hints de drag-and-drop) se marca con `data-pdf-exclude` en el JSX y se
// omite en la captura. Las imágenes nunca salen del browser: html2canvas trabaja
// sobre el DOM local y el PDF se arma en memoria antes de descargarse.

import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';

export interface ReportOptions {
  /** Nombre base del archivo (sin extensión). Default: informe-parecido-<timestamp>. */
  filename?: string;
}

/** ¿Este elemento se omite del PDF? (no esencial en un documento estático). */
function isExcluded(el: Element): boolean {
  return el instanceof HTMLElement && el.dataset.pdfExclude !== undefined;
}

// Rasteriza `element` con html2canvas y lo pagina en A4 (retrato) cortando la
// captura en franjas del alto de página, para no deformarla ni escalarla a una
// sola hoja cuando el informe es más alto que una página.
export async function generateReportPdf(
  element: HTMLElement,
  opts: ReportOptions = {},
): Promise<void> {
  const canvas = await html2canvas(element, {
    scale: 2,                  // texto/barras nítidos
    backgroundColor: '#ffffff',
    useCORS: true,
    logging: false,
    ignoreElements: isExcluded,
  });

  const pdf = new jsPDF({ unit: 'mm', format: 'a4' });
  const pageW = pdf.internal.pageSize.getWidth();   // 210
  const pageH = pdf.internal.pageSize.getHeight();  // 297
  const M = 10;
  const contentWmm = pageW - 2 * M;
  const contentHmm = pageH - 2 * M;

  // px de la captura que entran en una página manteniendo el ancho a contentWmm.
  const pxPerMm = canvas.width / contentWmm;
  const pageHpx = Math.max(1, Math.floor(contentHmm * pxPerMm));

  let renderedHpx = 0;
  let page = 0;
  while (renderedHpx < canvas.height) {
    const sliceHpx = Math.min(pageHpx, canvas.height - renderedHpx);
    const slice = document.createElement('canvas');
    slice.width = canvas.width;
    slice.height = sliceHpx;
    const ctx = slice.getContext('2d')!;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, slice.width, slice.height);
    ctx.drawImage(canvas, 0, renderedHpx, canvas.width, sliceHpx, 0, 0, canvas.width, sliceHpx);

    const sliceHmm = sliceHpx / pxPerMm;
    if (page > 0) pdf.addPage();
    // JPEG (no PNG): el fondo ya es blanco opaco, así que sin alpha; recorta el peso
    // ~10× (un PDF con 3 fotos full-DOM en PNG@2x pesa decenas de MB) sin perder
    // legibilidad de texto/barras a q=0.92.
    pdf.addImage(slice.toDataURL('image/jpeg', 0.92), 'JPEG', M, M, contentWmm, sliceHmm);

    renderedHpx += sliceHpx;
    page++;
  }

  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
  pdf.save(`${opts.filename ?? `informe-parecido-${stamp}`}.pdf`);
}
