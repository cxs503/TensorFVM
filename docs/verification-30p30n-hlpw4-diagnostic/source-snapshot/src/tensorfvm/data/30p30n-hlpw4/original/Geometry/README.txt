This geometry definition of the 30P-30N comes from the BANC series of workshops.
It is very close to the original 30P-30N shape.  The main exception is that the
BANC version has a thicker slat trailing edge.  See also AIAA-2014-2080 and
AIAA-2015-2844.

The current geometry (tested for primarily acoustic purposes at JAXA and at FSU)
had a nominal (stowed) chord of 18 inches.  (Note that the original 30P-30N tested in the 
NASA Langley LTPT had a larger chord of 22 inches.  However, this does not matter as
long as the Reynolds number is set correctly for the current set of test cases: 
i.e., 9 million based on stowed chord length.
In other words, when using the airfoil definition with 18 inch stowed chord, the
Reynolds number should be set to 500,000 per inch.)

For consistency, the farfield should be defined by a circle centered at (0,0) with radius
of 500c (9,000 inches when the airfoil stowed chord is 18 inches).

For scale-resolved simulations, the "2-D" computations should be performed on the
airfoil extruded a distance of 2 inches (11.11% of stowed chord) in the spanwise
direction.  Periodic boundary conditions should be applied on the spanwise surfaces.
Grid studies should be used to establish that you have sufficient resolution in the 
2D plane and in the spanwise direction.

Files provided (18 inch stowed chord):
2010_30p30n_thik_te_18inches.igs     - IGS CAD file
2010_30p30n_thik_te_18inches.stp     - STEP CAD file
2010_30p30n_thik_te_18inches.crv.dat - point definitions taken from the geometry
