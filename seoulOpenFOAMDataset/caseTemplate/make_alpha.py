#!/usr/bin/env python3
"""
0/alpha.water 초기조건: 지형 전체에 두께 FILM_DEPTH[m] 물막
(설계강우 등가 총 강우량에 해당) -> k=0 층에 부분충전.
"""

import sys

NX, NY, NZ = 96, 96, 10
DZ = 1.0 / NZ  # 0.1 m
FILM_DEPTH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.045  # m
ALPHA_K0 = min(FILM_DEPTH / DZ, 1.0)

n_cells = NX * NY * NZ

header = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      alpha.water;
}

dimensions      [0 0 0 0 0 0 0];

internalField   nonuniform List<scalar>
"""

with open("0/alpha.water", "w") as f:
    f.write(header)
    f.write(f"{n_cells}\n(\n")
    line_k0 = f"{ALPHA_K0:.4f}\n"
    line_0 = "0.0000\n"
    for k in range(NZ):
        block = (line_k0 if k == 0 else line_0) * (NX * NY)
        f.write(block)
    f.write(")\n;\n\n")
    f.write("""boundaryField
{
    terrain
    {
        type            zeroGradient;
    }
    atmosphere
    {
        type            inletOutlet;
        inletValue      uniform 0;
        value           uniform 0;
    }
    sides
    {
        type            inletOutlet;
        inletValue      uniform 0;
        value           uniform 0;
    }
}
""")

print(f"wrote 0/alpha.water: {n_cells} cells, film depth {FILM_DEPTH}m -> alpha(k=0)={ALPHA_K0:.4f}")
