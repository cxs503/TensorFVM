"""Recover vector-digitized reference facts, without copying the paper into reports."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET


def extract(html):
    result={}
    for figure,name,maximum in [('S6.F10','lift',1.4),('S6.F11','drag',1.)]:
        start=html.index('id="'+figure+'.pic1"');begin=html.rfind('<svg',0,start);end=html.index('</svg>',start)+6
        svg=html[begin:end];root=ET.fromstring(svg)
        candidates=[]
        for group in root.iter('g'):
            if group.get('stroke')!='#000000':continue
            for path in group.findall('path'):
                d=path.get('d','')
                if d.startswith('M 0 ') and d.count('L')>=20 and 'M' not in d[1:]:candidates.append(d)
        if len(candidates)!=1:raise ValueError('Ambiguous present-study vector curve')
        points=[tuple(map(float,p)) for p in re.findall(r'[ML]\s+([-\d.]+)\s+([-\d.]+)',candidates[0])]
        x,y=min(points,key=lambda p:abs(p[0]-286.43*4/30))
        if abs(x-286.43*4/30)>.006:raise ValueError('Missing four degree marker')
        result[name]=dict(figure=figure,series='present study, black line and filled circles',angle_degrees=4,svg_point=[x,y],axis_svg_height=190.95,axis_coefficient_maximum=maximum,value=y/190.95*maximum,absolute_digitization_uncertainty=.0001,figure_sha256=hashlib.sha256(svg.encode()).hexdigest())
    return dict(source_url='https://arxiv.org/html/2006.10487',source_html_sha256=hashlib.sha256(html.encode()).hexdigest(),method='Select black present-study long polyline; map alpha=4 marker using zero-origin axes. Coordinates rounded to 0.01 SVG units; +/-0.0001 covers extraction rounding, not author numerical uncertainty.',reference=result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--html',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    html=a.html.read_text() if a.html else urllib.request.urlopen('https://arxiv.org/html/2006.10487',timeout=60).read().decode()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(extract(html),indent=2)+'\n')
if __name__=='__main__':main()
