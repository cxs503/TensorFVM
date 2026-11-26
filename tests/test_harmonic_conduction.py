import numpy as np
import pytest

from tensorfvm.cht.energy import solve_harmonic_conduction


def test_harmonic_single_cell_complex_wall_and_source_power():
    omega, west, north, east, source = 3., 2.-1j, -.5+.8j, -.4j, .7+1.5j
    bc = {"west": {"type": "dirichlet", "values": west},
          "north": {"type": "robin", "values": north, "film_coefficient": 8.},
          "east": {"type": "neumann", "values": east}}
    t, wall = solve_harmonic_conduction([2.], [1.], 2., 3., bc, omega,
                                        source=source, conductivity_y=4.)
    # West conductance = 2, north film in series with half cell gives 8.
    exact = (2*west + 8*north + east + 2*source)/(10 + 6j*omega)
    np.testing.assert_allclose(t, exact)
    for side, power in dict(west=2*(west-exact), north=8*(north-exact), east=east, south=0.).items():
        np.testing.assert_allclose(wall[side], power)


def test_insulated_pair_contact_with_opposing_heat_inputs():
    t, walls = solve_harmonic_conduction([1.,1.], [1.], 2., 1., {}, 4., source=[1j,-1j],
                                         contact_resistance_x=.5)
    np.testing.assert_allclose(t[0], np.array([1j,-1j])/(2+4j))
    for power in walls.values():
        np.testing.assert_array_equal(power, 0.)
    with pytest.raises(ValueError, match="angular_frequency"):
        solve_harmonic_conduction([1.], [1.], 1., 1., {}, 0.)


def test_harmonic_cell_storage_matches_every_oriented_face_and_source():
    dx, dy = np.array([.2,.3,.5]), np.array([.3,.7])
    volume = np.outer(dy, dx)
    source = np.array([[1j,2.,.5],[-.6,1.2j,3.]])
    bc = {"west": {"type":"robin", "values":2.-1j, "film_coefficient":3.},
          "north": {"type":"neumann", "values":[1.,-1j,.4j]}}
    t, walls, power = solve_harmonic_conduction(dx, dy, [1.,2.,.7], 3., bc, 2.,
                      conductivity_y=4., contact_resistance_x=.2, contact_resistance_y=.3,
                      source=source, return_interior_power=True)
    x = np.column_stack((walls["west"],power["x"],-walls["east"]))
    y = np.vstack((walls["south"],power["y"],-walls["north"]))
    stored = 2j*3.*volume*t
    np.testing.assert_allclose(stored, source*volume-np.diff(x,axis=1)-np.diff(y,axis=0), atol=1e-13)


@pytest.mark.parametrize("dy", [[.7], [.3, .7], [.2, .3, .5]])
def test_periodic_y_seam_material_resistance_and_cell_balance(dy):
    nx, ny = 2, len(dy)
    dx = np.array([.2, .8])
    ky = 1+np.arange(ny*nx).reshape(ny, nx)
    volume = np.outer(dy, dx)
    source = 1j*ky+.3
    t, wall, interior = solve_harmonic_conduction(dx, dy, 3., 2., {}, 4., conductivity_y=ky,
                            source=source, return_interior_power=True, periodic_y=True)
    expected = dx*(t[-1]-t[0])/(.5*dy[-1]/ky[-1]+.5*dy[0]/ky[0])
    np.testing.assert_allclose(wall["south"], expected)
    np.testing.assert_array_equal(wall["south"], -wall["north"])
    qy = np.vstack((wall["south"], interior["y"], -wall["north"]))
    qx = np.column_stack((wall["west"], interior["x"], -wall["east"]))
    np.testing.assert_allclose(8j*volume*t, source*volume-np.diff(qy,axis=0)-np.diff(qx,axis=1))
    with pytest.raises(ValueError, match="periodic_y"):
        solve_harmonic_conduction(dx, dy, 1., 1., {"north": {"type":"neumann", "values":0.}},
                                  4., periodic_y=True)


@pytest.mark.parametrize("h", [0., 3.])
def test_dynamic_wall_matches_explicit_two_temperature_system(h):
    omega, cw, ambient, src = 4., .7, .3+.2j, 1.-.1j
    # Unit cube, unit depth: G_half=4, cell capacity=3.
    matrix = [[4+3j*omega,-4],[-4,4+h+1j*omega*cw]]
    cell, wall = np.linalg.solve(matrix,[src,h*ambient])
    data = dict(type="robin",values=ambient,film_coefficient=h,wall_areal_heat_capacity=cw)
    u, powers = solve_harmonic_conduction([1.],[1.],2.,3.,{"west":data},omega,source=src)
    np.testing.assert_allclose(u, cell)
    np.testing.assert_allclose(powers["west"], 4*(wall-cell))
    np.testing.assert_allclose(powers["west"], h*(ambient-wall)-1j*omega*cw*wall)


def test_mixed_face_storage_closes_individual_wall_balances():
    h, cw = np.array([2., 4., 0.]), np.array([1., 0., 0.])
    ambient = np.array([.5+.8j, -.2, .7j])
    dy = np.array([.2,.3,.5])
    west = dict(types=["robin","robin","neumann"], values=ambient,
                film_coefficient=h, wall_areal_heat_capacity=cw)
    u, powers = solve_harmonic_conduction([.4],dy,3.,2.,{"west":west},5.)
    flux = powers["west"]/dy
    surface = u[:,0]+flux*(.2/3.)
    np.testing.assert_allclose(flux[:2],(h*(ambient-surface)-5j*cw*surface)[:2])
    np.testing.assert_allclose(flux[2],ambient[2])
    np.testing.assert_allclose(5j*2.*.4*(dy @ u[:,0]),powers["west"].sum())
    west["wall_areal_heat_capacity"] = [0., 0., 1.]
    with pytest.raises(ValueError,match="Robin faces"):
        solve_harmonic_conduction([.4],dy,3.,2.,{"west":west},5.)


def test_low_frequency_storage_power_with_insulated_surface_capacity():
    from tensorfvm.cht.energy import solve_low_frequency_conduction
    dx = np.array([.2,.3,.5])
    drive = .7+.4j
    bc = {"west":dict(type="dirichlet",values=drive),
          "east":dict(type="robin",film_coefficient=0.,wall_areal_heat_capacity=.8,values=3.)}
    steady,lag = solve_low_frequency_conduction(dx,[1.],2.,3.,bc,return_interior_power=True)
    np.testing.assert_allclose(steady[0],drive)
    np.testing.assert_allclose(lag[1]["west"],drive*3.8)
    np.testing.assert_allclose(lag[1]["east"],-drive*.8)
    np.testing.assert_allclose(lag[2]["x"][0],drive*(.8+3*(1-np.cumsum(dx)[:-1])))


def test_unanchored_low_frequency_unbalanced_power_rejected():
    from tensorfvm.cht.energy import solve_low_frequency_conduction
    with pytest.raises(ValueError,match="zero net power"):
        solve_low_frequency_conduction([.3,.7],[1.],1.,1.,{},source=1.,periodic_y=True)


def test_closed_slab_storing_surface_has_zero_energy_at_both_orders():
    from tensorfvm.cht.energy import solve_low_frequency_conduction
    dx = np.array([.25,.25,.5])
    capacity = np.array([1.,3.,1.5])
    source = np.array([2.,0.,-1.])
    surface = .7
    bc = {"east":dict(type="robin",film_coefficient=0.,wall_areal_heat_capacity=surface,values=0.)}
    steady,lag = solve_low_frequency_conduction(dx,[1.],2.,capacity,bc,source=source)
    for field,power in (steady,lag):
        wall = field[0,-1]+power["east"][0]*dx[-1]/4.
        assert abs((capacity*dx)@field[0]+surface*wall)<1e-13
    actual,_ = solve_harmonic_conduction(dx,[1.],2.,capacity,bc,.01,source=source)
    np.testing.assert_allclose(actual,steady[0]+.01j*lag[0],rtol=0,atol=3e-7)


@pytest.mark.parametrize("vertical",[False,True])
def test_two_cell_wall_and_contact_complex_manufactured_solution(vertical):
    # Equal d²/k makes the two quadratic cell-mean corrections cancel at
    # the interior face even with a contact jump and unequal cell widths.
    widths = np.array([.25,.5])
    k = np.array([1.,4.])
    area, resistance, omega = .8,.27,3.
    west, q0, slope = 1.+.3j, .7-.4j, .35+.2j
    means, face_flux = [], [q0]
    surface = west
    for i,(d,material) in enumerate(zip(widths,k)):
        q = face_flux[-1]
        means.append(surface-(q*d/2+slope*d*d/6)/material)
        surface -= (q*d+slope*d*d/2)/material
        face_flux.append(q+slope*d)
        if i == 0:
            surface -= resistance*face_flux[-1]
    axis = "y" if vertical else "x"
    lo,hi = ("south","north") if vertical else ("west","east")
    bc = {lo:dict(type="dirichlet",values=west),hi:dict(type="dirichlet",values=surface)}
    shape = (2,1) if vertical else (1,2)
    expected = np.array(means).reshape(shape)
    dx,dy = ([area],widths) if vertical else (widths,[area])
    opts = {"contact_resistance_"+axis:resistance}
    temperature,powers,interior = solve_harmonic_conduction(dx,dy,1. if vertical else k,2.,bc,
        omega,source=slope+1j*omega*2*expected,conductivity_y=k[:,None] if vertical else None,
        dirichlet_flux_order=2,return_interior_power=True,**opts)
    np.testing.assert_allclose(temperature,expected,rtol=0.,atol=2e-14)
    np.testing.assert_allclose([powers[lo].item(),interior[axis].item(),-powers[hi].item()],
                               area*np.array(face_flux),rtol=0.,atol=2e-14)


def test_wall_flux_lag_matches_integrated_storage_across_contact():
    from tensorfvm.cht.energy import solve_low_frequency_conduction

    widths, k, resistance = np.array([.25,.5]),np.array([1.,4.]),.27
    bc = {side:dict(type="dirichlet",values=1.) for side in ("west","east")}
    dc,lag = solve_low_frequency_conduction(widths,[1.],k,1.,bc,
        contact_resistance_x=resistance,dirichlet_flux_order=2,return_interior_power=True)
    # At DC the field is 1 K. At first order q'=-1 and the two prescribed
    # wall temperatures have zero lag, so integral(q/k) + R*q(contact)=0.
    edges = np.r_[0.,np.cumsum(widths)]
    q0 = (np.sum(np.diff(edges**2)/(2*k))+resistance*edges[1])/(sum(widths/k)+resistance)
    np.testing.assert_allclose(dc[0],1.,atol=2e-14)
    np.testing.assert_allclose([lag[1]["west"].item(),lag[2]["x"].item(),-lag[1]["east"].item()],
                               q0-edges,rtol=0.,atol=2e-14)


@pytest.mark.parametrize("omega", [.03125, 1., 32.])
@pytest.mark.parametrize("film", [0., 3.])
def test_wall_heater_phasor_matches_two_capacity_balance(omega,film,record_property):
    heater,ambient = 1.+.75j, -.5+.25j
    wall = dict(type="robin",values=ambient,film_coefficient=film,
                wall_areal_heat_capacity=2.,wall_heater_flux=heater)
    t,power = solve_harmonic_conduction([1.],[.5],1.,1.,{"west":wall},omega)
    # Per unit area: s*T=2*(Tw-T), 2*s*Tw=P+h*(Ta-Tw)-2*(Tw-T).
    s = 1j*omega
    expected = (heater+film*ambient)/(s+(2*s+film)*(1+s/2))
    exact_power = .5*s*expected
    errors = dict(cell_phasor_K=abs(t.item()-expected),
                  material_power_phasor_W=abs(power["west"].item()-exact_power))
    for name,error in errors.items():
        record_property(name,float(error))
        assert error < 1e-13


def test_balanced_insulated_wall_heater_low_frequency_coefficients():
    from tensorfvm.cht.energy import solve_low_frequency_conduction

    heater = 1.+.5j
    bc = {"west":dict(type="robin",values=0.,film_coefficient=0.,wall_areal_heat_capacity=2.,
                      wall_heater_flux=heater)}
    dc,lag = solve_low_frequency_conduction([1.],[1.],1.,1.,bc,source=-heater)
    # Zero total stored energy fixes Tcell=-P/(3+s); Qwest=P+s*Tcell.
    np.testing.assert_allclose(dc[0],-heater/3)
    np.testing.assert_allclose(lag[0],heater/9)
    np.testing.assert_allclose(dc[1]["west"],heater)
    np.testing.assert_allclose(lag[1]["west"],-heater/3)


@pytest.mark.parametrize("omega", [.03125, 1., 32.])
@pytest.mark.parametrize("film", [0., 3.])
def test_harmonic_storing_wall_response(omega,film,record_property):
    heater,ambient = 1.+.75j,-.5+.25j
    spec = dict(type="robin",values=ambient,film_coefficient=film,
                wall_areal_heat_capacity=2.,wall_heater_flux=heater)
    _,powers,_,wall = solve_harmonic_conduction([.5],[1.],.25,1.,{"north":spec},omega,
        conductivity_y=1.,return_interior_power=True,return_storing_wall_response=True)
    s = 1j*omega
    cell = (heater+film*ambient)/(s+(2*s+film)*(1+s/2))
    surface = (1+s/2)*cell
    ref = dict(temperature=surface,heater=.5*heater,ambient=.5*film*(ambient-surface),
               material=.5*s*cell,storage=s*surface)
    response = wall["north"]
    for name, expected in ref.items():
        error = abs(response[name].item()-expected)
        record_property(name+"_error",float(error))
        assert error < 1e-13
    np.testing.assert_array_equal(response["material"],powers["north"])
    balance = abs((response["ambient"]+response["heater"]-response["material"]-response["storage"]).item())
    record_property("energy_balance_W",float(balance))
    assert balance < 1e-13


def test_harmonic_storing_wall_response_selects_mixed_faces_without_heater():
    spec = dict(types=["robin","neumann","robin"],values=[1.,-.25,2j],
                wall_areal_heat_capacity=[1.,0.,3.],film_coefficient=[2.,0.,4.])
    cell,power,wall = solve_harmonic_conduction([.5],[.125,.25,.375],2.,1.,{"west":spec},3.,
                                               return_storing_wall_response=True)
    report = wall["west"]
    assert set(wall)=={"west"} and report["temperature"].shape==(2,)
    np.testing.assert_array_equal(report["heater"],np.zeros(2))
    rows = [0,2]
    area = np.array([.125,.375])
    np.testing.assert_allclose(report["temperature"],cell[rows,0]+.125*power["west"][rows]/area)
    np.testing.assert_allclose(report["storage"],report["ambient"]-report["material"])


@pytest.mark.parametrize("film", [0., 3., 1e9])
def test_low_frequency_storing_wall_surface_coefficients(film,record_property):
    from tensorfvm.cht.energy import solve_low_frequency_conduction

    area, heater, ambient = .5, 1.+.75j, -.5+.25j
    wall = dict(type="robin",values=ambient,film_coefficient=film,
                wall_areal_heat_capacity=2.,wall_heater_flux=heater)
    # Surface balance with the cell eliminated:
    # [h+2s+s/(1+s/2)]Tw = P+h*Ta+S/(1+s/2).
    # For the floating system S=-P and the conserved total energy is zero.
    source = -heater if film==0 else 0.
    dc,lag = solve_low_frequency_conduction([.5],[1.],.25,1.,{"north":wall},
        conductivity_y=1.,source=source,return_interior_power=True,return_storing_wall_response=True)
    if film:
        w0 = ambient+heater/film
        w1 = -3*w0/film
    else:
        w0,w1 = heater/6,-heater/18
    q0 = -source
    q1 = w0-q0/2
    for order,(out,ws,q,store,drive,air) in enumerate((
            (dc,w0,q0,0.,heater,-heater if film else 0.),
            (lag,w1,q1,2*w0,0.,-film*w1))):
        result = out[-1]["north"]
        expected = dict(temperature=ws,heater=area*drive,ambient=area*air,
                        material=area*q,storage=area*store)
        for name,value in expected.items():
            error = abs(result[name].item()-value)
            record_property(f"order_{order}_{name}_error",float(error))
            assert error < 3e-13


@pytest.mark.parametrize("source,heater", [(0.,1.+.5j),(.25-.5j,0.),(.25j,.5j)])
def test_floating_plate_uniform_rate_and_regular_surface(source,heater,record_property):
    from tensorfvm.cht.energy import solve_low_frequency_conduction

    # Per unit area, s*T=S+2*(Tw-T), 2*s*Tw=P-2*(Tw-T).
    # Thus T+2Tw=(S+P)/s and Tw-T=(P/2-S)/(3+s).
    wall = dict(type="robin",film_coefficient=0.,wall_areal_heat_capacity=2.,wall_heater_flux=heater)
    rate,dc,first = solve_low_frequency_conduction([1.],[.5],1.,1.,{"west":wall},source=source,
        return_uniform_rate=True,return_storing_wall_response=True)
    exact_rate = (source+heater)/3
    difference = (heater/2-source)/3
    errors = dict(rate_K_per_s=abs(rate-exact_rate))
    for j,(result,delta) in enumerate(((dc,difference),(first,-difference/3))):
        response = result[-1]["west"]
        errors[f"order_{j}_cell"] = abs(result[0].item()+2*delta/3)
        errors[f"order_{j}_surface"] = abs(response["temperature"].item()-delta/3)
        errors[f"order_{j}_material"] = abs(response["material"].item()-delta)
        errors[f"order_{j}_storage"] = abs(response["storage"].item()-(exact_rate if j==0 else difference/3))
        errors[f"order_{j}_balance"] = abs((response["heater"]-response["material"]-response["storage"]).item())
    for name,error in errors.items():
        record_property(name,float(error))
        assert error < 2e-14


def test_uniform_rate_for_plate_without_wall_capacity_and_for_anchored_plate():
    from tensorfvm.cht.energy import solve_low_frequency_conduction

    opts = dict(dx=[.25,.75],dy=[.5],conductivity=1.,volumetric_heat_capacity=[1.,2.],source=[2.,4.])
    rate,dc,lag = solve_low_frequency_conduction(**opts,bc={},return_uniform_rate=True)
    assert rate==2.
    np.testing.assert_allclose(dc[0],0.,atol=1e-15)
    np.testing.assert_allclose(lag[0],0.,atol=1e-15)
    bc = {"east":dict(type="dirichlet",values=0.)}
    rate,dc,lag = solve_low_frequency_conduction(**opts,bc=bc,return_uniform_rate=True)
    assert rate==0.
    ref = solve_low_frequency_conduction(**opts,bc=bc)
    np.testing.assert_array_equal(dc[0],ref[0][0])
    np.testing.assert_array_equal(lag[0],ref[1][0])


@pytest.mark.parametrize("omega", [1e-16, 1e-8, 1., 32.])
@pytest.mark.parametrize("source,heater", [(0.,1.+.5j),(.25-.5j,0.),(.25j,.5j)])
def test_floating_harmonic_plate_resolves_relative_temperature(omega,source,heater,record_property):
    # Exact at every frequency: s*(T+2Tw)=S+P, (s+3)*(Tw-T)=P/2-S.
    s = 1j*omega
    spec = dict(type="robin",film_coefficient=0.,wall_areal_heat_capacity=2.,wall_heater_flux=heater)
    rate,t,power,response = solve_harmonic_conduction([1.],[.5],1.,1.,{"west":spec},omega,
        source=source,return_uniform_rate=True,return_storing_wall_response=True)
    diff = (heater/2-source)/(3+s)
    wall = response["west"]
    errors = dict(rate_K_s=abs(rate-(source+heater)/3),cell_K=abs(t.item()+2*diff/3),
                  surface_K=abs(wall["temperature"].item()-diff/3),material_W=abs(power["west"].item()-diff),
                  storage_W=abs(wall["storage"].item()-(rate+s*diff/3)),
                  total_excess_energy_J=abs(.5*t.item()+wall["temperature"].item()))
    for name,error in errors.items():
        record_property(name,float(error))
        assert error < 1e-14


def test_uniform_rate_multicell_periodic_plate_with_flux_load():
    # Uniform source plus a boundary flux: the relative field must satisfy
    # every local balance and have zero capacity-weighted mean.
    area = np.outer([.25,.75],[.5,.5])
    mass = area*np.array([1.,2.])
    flux = np.array([1.,-.25])
    bc = {"east":dict(type="neumann",values=flux)}
    opts = dict(dx=[.5,.5],dy=[.25,.75],conductivity=2.,volumetric_heat_capacity=[1.,2.],
                source=.75,periodic_y=True,return_uniform_rate=True,return_interior_power=True)
    rate,t,power,cuts = solve_harmonic_conduction(**opts,bc=bc,angular_frequency=1e-15)
    np.testing.assert_allclose(rate,(.75+np.dot([.25,.75],flux))/mass.sum())
    np.testing.assert_allclose((mass*t).sum(),0.,atol=1e-15)
    net = np.column_stack((-cuts["x"][:,0],cuts["x"][:,0]+power["east"]))
    net[0] += power["south"]-cuts["y"][0]
    net[1] += cuts["y"][0]+power["north"]
    np.testing.assert_allclose(mass*(rate+1e-15j*t),.75*area+net,atol=1e-14)
    rate,*_ = solve_harmonic_conduction(**opts,bc={"west":dict(type="dirichlet",values=1.)},angular_frequency=1.)
    assert rate==0.


@pytest.mark.parametrize("omega", [.0625, 1., 16.])
def test_harmonic_cell_power_with_independent_bulk_ambient(omega,record_property):
    # Two unequal cells exchange heat across a finite contact. Invert the
    # two energy equations analytically and check each physical channel.
    dx,dy = np.array([.25,.75]),.5
    volume = dx*dy
    mass = volume*np.array([2.,3.])
    bulk = volume*np.array([.75,2.])
    source = np.array([1.+.25j,-.125+.75j])
    ambient = np.array([.25-.5j,1.+.25j])
    g = dy/(.125+.25+.375/3)
    a,b = g+bulk+1j*omega*mass
    rhs = source*volume+bulk*ambient
    expected = np.array([b*rhs[0]+g*rhs[1],g*rhs[0]+a*rhs[1]])/(a*b-g*g)
    t,_,cut,cell = solve_harmonic_conduction(dx,[dy],[1.,3.],[2.,3.],{},omega,
        contact_resistance_x=.25,source=source,heat_loss_coefficient=[.75,2.],
        bulk_ambient_temperature=ambient,return_cell_power=True,return_interior_power=True)
    q = cut["x"].item()
    errors = dict(temperature_K=np.max(abs(t-expected)),source_W=np.max(abs(cell["source"]-source*volume)),
                  bulk_W=np.max(abs(cell["bulk"]-bulk*(ambient-expected))),
                  storage_W=np.max(abs(cell["storage"]-1j*omega*mass*expected)),
                  interface_W=abs(q-g*(expected[0]-expected[1])),
                  balance_W=np.max(abs(cell["storage"]-cell["source"]-cell["bulk"]-np.array([-q,q]))))
    for name,error in errors.items():
        record_property(name,float(error))
        assert error < 2e-14


def test_harmonic_cell_storage_retains_common_heating_power():
    rate,t,_,cell = solve_harmonic_conduction([.25,.75],[1.],1.,[1.,2.],{},1e-16,
        source=[1.,2.],return_uniform_rate=True,return_cell_power=True)
    assert rate==1.
    np.testing.assert_allclose(t,0.,atol=1e-15)
    np.testing.assert_array_equal(cell["source"],cell["storage"])
    np.testing.assert_array_equal(cell["bulk"],0.)
