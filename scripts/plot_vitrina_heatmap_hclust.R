#!/usr/bin/env Rscript

# Heatmaps clusterizados de la vitrina 2026 usando R base.
#
# Entrada esperada:
#   data/output/teams/vitrina_transfermarkt_northamerica2026_similarity_pilot.json
#
# Metodo:
#   - matriz original = similitud agregada entre selecciones (mean o median).
#   - distancia = 1 - similitud.
#   - clustering = stats::hclust() con varios metodos.
#   - visual = heatmap() de R base, reordenando filas/columnas por el mismo arbol.

args <- commandArgs(trailingOnly = TRUE)

input <- if (length(args) >= 1) {
  args[[1]]
} else {
  "data/output/teams/vitrina_transfermarkt_northamerica2026_similarity_pilot.json"
}

out_dir <- if (length(args) >= 2) {
  args[[2]]
} else {
  "data/output/teams/r_heatmaps"
}

aggregation_arg <- if (length(args) >= 3) {
  args[[3]]
} else {
  "all"
}

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Falta el paquete R 'jsonlite'. Instalar con: install.packages('jsonlite')")
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

payload <- jsonlite::fromJSON(input, simplifyVector = FALSE)
teams <- unlist(payload$teams, use.names = FALSE)
methods <- c("average", "complete", "ward.D2", "single")

matrix_from_payload <- function(aggregation) {
  source <- payload$team_similarity_matrix
  if (!is.null(payload$team_similarity_matrices[[aggregation]])) {
    source <- payload$team_similarity_matrices[[aggregation]]
  }
  sim <- do.call(rbind, lapply(source, function(row) {
    as.numeric(unlist(row, use.names = FALSE))
  }))
  rownames(sim) <- teams
  colnames(sim) <- teams
  sim
}

plot_one_aggregation <- function(aggregation) {
  agg_dir <- file.path(out_dir, aggregation)
  dir.create(agg_dir, recursive = TRUE, showWarnings = FALSE)

  sim <- matrix_from_payload(aggregation)

  # La diagonal del payload representa similitud intra-seleccion; para clusterizar
  # selecciones conviene fijarla en 1.0 como identidad geométrica de la distancia.
  diag(sim) <- 1

  dist_mat <- 1 - sim
  diag(dist_mat) <- 0
  dist_mat[!is.finite(dist_mat)] <- max(dist_mat[is.finite(dist_mat)], na.rm = TRUE)

  write.csv(sim, file.path(agg_dir, paste0("team_similarity_matrix_", aggregation, ".csv")))
  write.csv(dist_mat, file.path(agg_dir, paste0("team_distance_1_minus_similarity_", aggregation, ".csv")))

  for (method in methods) {
    hc <- hclust(as.dist(dist_mat), method = method)
    ordered <- hc$labels[hc$order]

    png(
      filename = file.path(agg_dir, paste0("heatmap_", aggregation, "_hclust_", method, ".png")),
      width = 1800,
      height = 1800,
      res = 160
    )
    op <- par(no.readonly = TRUE)
    par(mar = c(8, 8, 4, 3))
    heatmap(
      sim,
      Rowv = as.dendrogram(hc),
      Colv = as.dendrogram(hc),
      symm = TRUE,
      scale = "none",
      col = colorRampPalette(c("#3b0f70", "#2196f3", "#f9d423", "#d7191c"))(96),
      margins = c(9, 9),
      main = paste("Vitrina 2026 -", aggregation, "- hclust", method, "- distancia = 1 - similitud"),
      xlab = "",
      ylab = ""
    )
    par(op)
    dev.off()

    writeLines(
      ordered,
      con = file.path(agg_dir, paste0("order_", aggregation, "_hclust_", method, ".txt"))
    )
  }
}

aggregations <- if (aggregation_arg %in% c("all", "both")) {
  c("mean", "median", "top3_mean", "top5_mean")
} else {
  c(aggregation_arg)
}
for (aggregation in aggregations) {
  plot_one_aggregation(aggregation)
}

cat("Escritos heatmaps en:", normalizePath(out_dir), "\n")
cat("Metodos hclust:", paste(methods, collapse = ", "), "\n")
cat("Agregaciones:", paste(aggregations, collapse = ", "), "\n")
cat("Distancia usada: 1 - similitud agregada entre selecciones\n")
