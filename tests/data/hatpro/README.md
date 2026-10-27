# RPG input pairs

These are small subsets of measured binary files and their independent RPG
NetCDF exports. `provenance.json` records original hashes, format codes and
selected record indices. Original source files are never changed.

- `legacy`: Munich, 2020-06-15. BRT version 1, IRT version 2.
- `modern`: Munich, 2025-10-06. BRT version 2, IRT version 3.
- Both include BLB version 2 and MET with wind and rain-rate fields.

Each binary keeps its original header, with only the record count changed.
Selected records retain their original bytes. NetCDF references are sliced at
the same indices, retaining the manufacturer's variables and metadata. The
selection includes initial samples and samples near the first three scans.

The exports include a 0-degree surface-temperature channel absent from binary
BLB brightness temperatures. Binary BLB has no explicit azimuth field. Daily
import comparisons cover the common observations after the shared handling of
coordinates, time matching and averaging. The legacy export's frequency field
contains TB minima. The existing daily importer's standard-frequency correction
is retained and applies before comparison.

These pairs verify the represented formats. They do not establish support for
every historical instrument format. Failure tests modify temporary copies to
exercise truncated records, unsupported versions and explicit local-time offsets.
