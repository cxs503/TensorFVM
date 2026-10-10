"""Exact CPU transient state continuation and online statistical moments."""
from dataclasses import asdict
import json
import numpy as np
import torch

class FieldMoments:
    def __init__(self, solver, discard_time=100.):
        self.discard_time=float(discard_time)
        self.count=0;self.velocity=torch.zeros_like(solver.velocity);self.pressure=torch.zeros_like(solver.pressure)
        self.velocity_product=torch.zeros((solver.N,3,3),dtype=torch.float64)
    def add(self, solver):
        self.count+=1;n=self.count
        self.velocity+=(solver.velocity-self.velocity)/n
        self.pressure+=(solver.pressure-self.pressure)/n
        product=solver.velocity[:,:,None]*solver.velocity[:,None,:]
        self.velocity_product+=(product-self.velocity_product)/n
    @property
    def reynolds_stress(self):return self.velocity_product-self.velocity[:,:,None]*self.velocity[:,None,:]

def save_checkpoint(path,solver,moments,source_sha256):
    tmp=path.with_name(path.stem+'-next.npz')
    np.savez_compressed(tmp,velocity=solver.velocity.numpy(),pressure=solver.pressure.numpy(),flux=solver.flux.numpy(),time=solver.time,
        config_json=json.dumps(asdict(solver.config)),settings_json=json.dumps(asdict(solver.settings)),
        history_json=json.dumps(solver.history),forces_json=json.dumps(solver.forces),source_json=json.dumps(source_sha256),
        mean_discard_time=moments.discard_time,mean_velocity=moments.velocity.numpy(),mean_pressure=moments.pressure.numpy(),mean_velocity_product=moments.velocity_product.numpy(),mean_count=moments.count,
        centers=solver.centers.numpy(),volumes=solver.volumes.numpy(),owner=solver.owner.numpy(),neighbor=solver.neighbor.numpy(),area=solver.area.numpy(),face_centers=solver.fc.numpy(),
        vertices=solver.mesh.vertices.numpy(),connectivity=solver.mesh.connectivity.numpy(),wall=solver.wall.numpy(),outlet=solver.outlet.numpy())
    tmp.replace(path)

def restore_checkpoint(path,solver,moments,source_sha256):
    with np.load(path,allow_pickle=False)as z:
        prior=json.loads(str(z['config_json']));now=asdict(solver.config)
        if float(z['mean_discard_time'])!=moments.discard_time:raise ValueError('statistical discard policy mismatch')
        # max_steps is a launch limit and may increase upon continuation.
        prior.pop('max_steps');now.pop('max_steps')
        if prior!=now or json.loads(str(z['settings_json']))!=asdict(solver.settings):raise ValueError('checkpoint configuration mismatch')
        if json.loads(str(z['source_json']))!=source_sha256:raise ValueError('checkpoint producer source mismatch')
        for key in ('velocity','pressure','flux'):setattr(solver,key,torch.from_numpy(z[key].copy()))
        solver.time=float(z['time']);solver.history=json.loads(str(z['history_json']));solver.forces=json.loads(str(z['forces_json']))
        moments.count=int(z['mean_count']);moments.velocity=torch.from_numpy(z['mean_velocity'].copy());moments.pressure=torch.from_numpy(z['mean_pressure'].copy());moments.velocity_product=torch.from_numpy(z['mean_velocity_product'].copy())
