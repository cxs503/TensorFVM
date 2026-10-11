# Plot plane and field interpretation

The main native mesh/pressure/speed figure uses **z/L = 0.03**, explicitly stored in summary.json and the image title. The manufactured analytic pressure is rho U² sin(kx) cos(ky) sin(kz), which is identically zero at z=0; an offset section provides a nonzero analytic pressure background.

Each polygon is the true intersection of a native cell with this plane. Its colour is the actual solved cell-centroid value, with no smoothing or reconstructed plane evaluation. An intersected tetrahedron centroid may lie away from the plane; visible variation between polygons therefore includes cell-centroid sampling of the varying analytic field. It is not by itself the pressure discretization error. Full-volume centroid errors and actual current-equation residuals were independently verified in ../audit/audit.json.

This is a manufactured discretization verification with nonzero analytic hull boundary velocity. It is not a physical SUBOFF velocity/pressure field or experimental Cp validation.
