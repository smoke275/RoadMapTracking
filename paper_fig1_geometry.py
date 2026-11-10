"""Derived regions of Fig. 1 of the paper (RoadmapBasedTracking/figures/geometry_terms.tex): wedge, A(c), hidden part, shadow, entry nodes.
Run inside the roadmaptracking Docker image from the RoadmapTracking root:
  docker run --rm -v "$(pwd)":/app -w /app -e PYTHONPATH=/app roadmaptracking python paper_fig1_geometry.py
Paste the printed paths into the TikZ figure."""
import sys, math
sys.path.insert(0, '/app')
import visilibity as vis
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.ops import unary_union
from geometry import poly_to_points

# ccw, jagged, in the style of the benchmark polygons
V = [(0.40,-0.30),(1.60,0.15),(2.30,-0.45),(3.05,-0.05),(3.35,1.75),(4.05,1.55),(3.95,0.10),
     (5.20,-0.35),(6.40,0.30),(7.10,-0.55),(8.60,0.05),(9.40,-0.30),(10.40,0.45),(9.70,1.40),
     (10.30,2.20),(9.95,2.95),(6.00,3.00),(6.25,4.70),(7.60,4.40),(8.90,4.95),(10.15,4.55),
     (9.70,5.70),(8.30,5.35),(6.90,5.95),(5.10,5.45),(3.60,5.85),(2.20,5.30),(0.90,5.80),
     (0.10,5.20),(0.55,3.90),(-0.15,2.60),(0.35,1.30)]
P = Polygon(V)
assert P.is_valid and P.exterior.is_ccw, (P.is_valid, P.exterior.is_ccw)
ic = V.index((6.00,3.00)); c = V[ic]; vm = V[ic-1]; vp = V[(ic+1)%len(V)]
print('% c', c, 'v_{i-1}', vm, 'v_{i+1}', vp)
# reflex check
def cross(a,b,cc): return (b[0]-a[0])*(cc[1]-a[1])-(b[1]-a[1])*(cc[0]-a[0])
print('% reflex at c:', cross(vm,c,vp) < 0)
for i in range(len(V)):
    if cross(V[i-1],V[i],V[(i+1)%len(V)])<0: print('% reflex vertex', i, V[i])

def unit(d): n=math.hypot(*d); return (d[0]/n, d[1]/n)
u1 = unit((c[0]-vm[0], c[1]-vm[1])); u2 = unit((c[0]-vp[0], c[1]-vp[1]))
R = 40.0
wedge_tri = Polygon([c, (c[0]+R*u1[0], c[1]+R*u1[1]), (c[0]+R*u2[0], c[1]+R*u2[1])])
wedgeP = wedge_tri.intersection(P)

walls = vis.Polygon([vis.Point(float(x), float(y)) for x, y in V])
if walls.area() < 0: walls.reverse()
env = vis.Environment([walls]); env.PRINTING_DEBUG_DATA = False
# observer: c nudged into the wedge
obs = (c[0] + 0.004*(u1[0]+u2[0]), c[1] + 0.004*(u1[1]+u2[1]))
vp_c = vis.Visibility_Polygon(vis.Point(*obs), env, 1e-6)
VisC = Polygon([(vp_c[i].x(), vp_c[i].y()) for i in range(vp_c.n())]).buffer(0)
A = wedgeP.intersection(VisC)
hidden = wedgeP.difference(VisC)

# wedge ray hits on the boundary
def first_hit(origin, d):
    ray = LineString([origin, (origin[0]+R*d[0], origin[1]+R*d[1])])
    inter = ray.intersection(P.exterior)
    pts = [g for g in getattr(inter, 'geoms', [inter]) if not g.is_empty]
    cand = []
    for g in pts:
        for q in (g.coords if hasattr(g,'coords') else []):
            dist = math.hypot(q[0]-origin[0], q[1]-origin[1])
            if dist > 1e-6: cand.append((dist, q))
    return min(cand)[1]
h1 = first_hit(c, u1); h2 = first_hit(c, u2)
print('% ray1 hit', h1, 'ray2 hit', h2)

# roadmap and pursuer
k1=(1.5,5.1); k2=(1.5,1.1); k3=(8.5,1.5); pil_r=(4.05,1.55); pil_l=(3.35,1.75)
edges = [LineString([k1,k2]), LineString([k1,c]), LineString([c,k3]), LineString([k3,pil_r]),
         LineString([pil_r,pil_l]), LineString([pil_l,k2])]
for e in edges: assert P.buffer(1e-9).covers(e), e
ray1 = LineString([c,h1]); ray2 = LineString([c,h2])
for e in edges:
    for r,name in ((ray1,'ray1'),(ray2,'ray2')):
        x = e.intersection(r)
        if not x.is_empty and x.geom_type=='Point' and x.distance(Point(c))>1e-6:
            print('% entry node', name, (round(x.x,3), round(x.y,3)))
p=(1.5,4.3); e_=(9.3,0.5)
assert P.covers(Point(e_)) and P.covers(Point(p))
# geodesic e->c straight?
print('% e->c straight:', P.buffer(1e-9).covers(LineString([e_,c])))
# shadow: ray p->c beyond c to the wall
d = unit((c[0]-p[0], c[1]-p[1])); hs = first_hit(c, d)
print('% shadow hit', hs)
# shadow region: part of P not visible from p, adjacent to v_{i-1} side
shadow = Polygon([c, hs, (10.30,2.20), vm]).intersection(P)
hidden = hidden.buffer(-0.01).buffer(0.01)
A = A.buffer(-0.005).buffer(0.005)
print('% shadow area', round(shadow.area,3))

def tikz(geom, style):
    gs = list(geom.geoms) if hasattr(geom,'geoms') else [geom]
    out=[]
    for g in gs:
        if g.is_empty or g.area < 1e-6: continue
        g = g.simplify(1e-4)
        coords = list(g.exterior.coords)[:-1]
        out.append('\\fill[%s] ' % style + ' -- '.join('(%.3f,%.3f)'%(x,y) for x,y in coords) + ' -- cycle;')
    return '\n'.join(out)
print('% ---- polygon outline path')
print(' -- '.join('(%.2f,%.2f)'%(x,y) for x,y in V) + ' -- cycle')
print('% ---- wedge')
print(tikz(wedgeP, 'blue!8'))
print('% ---- A(c)')
print(tikz(A, 'blue!24'))
print('% ---- hidden')
print(tikz(hidden, 'pattern=north east lines, pattern color=black!55'))
print('% ---- shadow')
print(tikz(shadow, 'pattern=north west lines, pattern color=red!70!black'))
print('% u1 arrow end', (round(c[0]+1.7*u1[0],3), round(c[1]+1.7*u1[1],3)), 'u2 arrow end', (round(c[0]+1.4*u2[0],3), round(c[1]+1.4*u2[1],3)))
print('% angles u1 {:.2f} u2 {:.2f}'.format(math.degrees(math.atan2(u1[1],u1[0]))%360, math.degrees(math.atan2(u2[1],u2[0]))%360))
a1=math.radians(math.degrees(math.atan2(u1[1],u1[0]))%360); print('% arc start', (round(c[0]+1.1*math.cos(a1),3), round(c[1]+1.1*math.sin(a1),3)))
