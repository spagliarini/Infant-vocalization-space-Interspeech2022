# Infant-vocalization-space-Interspeech2022

## Pre-processing
1. Extraction of single vocalizations using MATLAB routine. Vocalizations are extracted based on the automatic LENA system. References:
> J. Gilkerson and J. A. Richards, “The lena natural language study,” Boulder, CO: LENA Foundation. Retrieved March, vol. 3, p. 2009, 2008. 
> D. Xu, U. Yapanel, S. Gray, J. Gilkerson, J. Richards, and J. Hansen, “Signal processing for young child speech language development,” in First Workshop on Child, Computer and Interaction, 2008.
3. Extraction of MFCCs coefficients, velocity and acceleration using OpenSMILE. Resources and guidelines can be found at https://audeering.github.io/opensmile/. References:
> Florian Eyben, Martin Wöllmer, Björn Schuller: "openSMILE - The Munich Versatile and Fast Open-Source Audio Feature Extractor", Proc. ACM Multimedia (MM), ACM, Florence, Italy, ISBN 978-1-60558-933-6, pp. 1459-1462, 25.-29.10.2010.

## Vocalizations space and statistical analysis
1. Compute the low-dimensional representation (UMPA, t-SNE, PCA): python code.
2. Compute statistical properties of the clusters across families and across time (age): python and R codes.
