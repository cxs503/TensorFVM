"""Losslessly share duplicated PDF resources; preserve page count and images."""
import argparse,json,tempfile
from pathlib import Path
from pypdf import PdfReader,PdfWriter,__version__
from pypdf.generic import EncodedStreamObject,DictionaryObject
from tensorfvm.verification.core import write_json,sha,artifact_manifest

p=argparse.ArgumentParser();p.add_argument('outputs',nargs='+',type=Path);a=p.parse_args()
for output in a.outputs:
 path=output/'report.pdf';reader=PdfReader(path);count=len(reader.pages);writer=PdfWriter();writer.append(reader);# Exact encoded bytes plus the complete stream dictionary are sufficient
 # for equality; avoid repeatedly decoding PNG predictors just to hash copies.
 original_hash=EncodedStreamObject.hash_value_data
 try:
  EncodedStreamObject.hash_value_data=lambda obj: DictionaryObject.hash_value_data(obj)+obj._data
  for _ in range(3):writer.compress_identical_objects(remove_duplicates=True,remove_unreferenced=True)
 finally:EncodedStreamObject.hash_value_data=original_hash
 with tempfile.TemporaryDirectory() as td:
  result=Path(td)/'report.pdf';writer.write(result)
  if len(PdfReader(result).pages)!=count:raise ValueError('PDF page count changed')
  before=path.stat().st_size;path.write_bytes(result.read_bytes())
 m=json.loads((output/'manifest.json').read_text());m['source_sha256']['scripts/compact_external_pdf.py']=sha(Path(__file__));m['publication_pypdf_version']=__version__;m['artifacts_sha256']=artifact_manifest(output);write_json(output/'manifest.json',m);print(output,'PDF bytes',before,'->',path.stat().st_size,'pages',count)
