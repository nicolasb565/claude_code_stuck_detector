You are implementing the `is_valid` algorithm for 3D polyhedral surfaces in Boost.Geometry.

A polyhedral surface is a composite geometry made of polygonal faces sharing edges (like a cube or tetrahedron). The validation must enforce OGC Simple Features rules:

1. **Contiguity** — all patches connect via shared boundaries (the surface is one connected component)
2. **Shared edges** — each boundary segment appears in at most 2 polygons
3. **Consistent orientation** — adjacent faces traverse shared edges in opposite directions (outward-facing normals)

The code is at /scratch

Implement the validator in:
- include/boost/geometry/algorithms/detail/is_valid/polyhedral_surface.hpp (new file)

You'll need to:
1. Build a face adjacency graph from shared edges
2. Check contiguity via biconnected component detection
3. Verify consistent orientation by checking edge traversal directions
4. Add new validity_failure_type enum values in algorithms/validity_failure_type.hpp
5. Wire the dispatch in algorithms/detail/is_valid/implementation.hpp

Follow the existing patterns for multi_polygon validation in the is_valid directory.

The full Boost tree is at /scratch. Build and run tests:
```bash
cd /scratch
./b2 libs/geometry/test --build-dir=build -j$(nproc)
```
