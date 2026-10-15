#!/bin/bash
# Rebuild GL/X stub libraries so the gmsh wheel loads headless. Usage: make_stubs.sh
set -e
LIB=$(find /home/user/.venv-tfvm -name "libgmsh.so*" | head -1)
OUT=/home/user/.stublibs; mkdir -p $OUT; cd $OUT
MISSING=$(ldd $LIB | grep "not found" | awk '{print $1}')
nm -D --undefined-only $LIB | awk '{print $2}' | sed 's/@.*//' | sort -u > undef.txt
: > defined.txt
for l in $(ldd $LIB | awk '/=> \//{print $3}'); do nm -D --defined-only $l 2>/dev/null | awk '{print $3}' | sed 's/@.*//' >> defined.txt; done
nm -D --defined-only $LIB | awk '{print $3}' | sed 's/@.*//' >> defined.txt
sort -u defined.txt -o defined.txt
comm -23 undef.txt defined.txt > tostub.txt
echo "stubbing $(wc -l < tostub.txt) symbols into libGL.so.1; empty stubs: $MISSING"
{ echo "/* auto-generated stubs */"; while read s; do echo "void $s(void){}"; done < tostub.txt; } > stub_all.c
echo "void __stub_empty(void){}" > stub_empty.c
for m in $MISSING; do
  if [ "$m" = "libGL.so.1" ]; then src=stub_all.c; else src=stub_empty.c; fi
  gcc -shared -fPIC -nostdlib -Wl,-soname,$m -o $m $src
done
LD_LIBRARY_PATH=$OUT /home/user/.venv-tfvm/bin/python -c "import gmsh;gmsh.initialize();print('gmsh',gmsh.__version__,'ok');gmsh.finalize()"
