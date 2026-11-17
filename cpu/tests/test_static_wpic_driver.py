"""Verify isolated WPIC physics selection, conserved mass, and exact continuation."""
import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

CPU = Path(__file__).resolve().parents[1]


@unittest.skipUnless((CPU/'venus_cpu_vlasov_poisson_wpic_static').exists() and
                     (CPU/'venus_cpu_vlasov_poisson_wpic_expanding').exists(),
                     'Build the two VP particle executables first.')
class StaticWpicTests(unittest.TestCase):
    def test_static_selection_and_restart(self):
        common=['--radial-cells','16','--radial-minimum','0.1','--radial-maximum','10',
                '--particles-per-radial-cell','128','--maximum-timestep','0.001',
                '--output-interval','0.001','--history-every','1','--restart-every','1','--threads','2']
        with tempfile.TemporaryDirectory(prefix='venus-static-wpic-') as name:
            root=Path(name)
            def run(binary,directory,end,extra=()):
                return subprocess.run([str(CPU/binary),*common,'--output-directory',str(directory),
                                       '--end-time',str(end),*extra],capture_output=True,text=True,timeout=30)
            full=root/'full';half=root/'half';continued=root/'continued';reference=root/'reference'
            for directory,end,extra in [(full,.004,()),(half,.002,()),
                    (continued,.004,('--restart',str(half/'restart_state.bin')))]:
                result=run('venus_cpu_vlasov_poisson_wpic_static',directory,end,extra)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=run('venus_cpu_vlasov_poisson_wpic_expanding',reference,.004,('--vp-universe','static'))
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            state=(full/'restart_state.bin').read_bytes()
            self.assertEqual(state,(continued/'restart_state.bin').read_bytes())
            self.assertEqual(state,(reference/'restart_state.bin').read_bytes())
            metadata=json.loads((full/'metadata.json').read_text())
            self.assertEqual(metadata['vp_universe'],'static')
            self.assertEqual(metadata['vp_alpha'],0)
            self.assertEqual(metadata['vp_poisson_source'],'total_density')
            with (full/'history.csv').open() as stream:rows=list(csv.DictReader(stream))
            mass=float(rows[0]['mobile_mass'])
            self.assertLess(max(abs(float(row['mobile_mass'])-mass) for row in rows),1e-12)
            for extra in [('--vp-universe','expanding'),('--subhalo-collision-strength','1')]:
                invalid=root/'invalid'
                result=run('venus_cpu_vlasov_poisson_wpic_static',invalid,.001,extra)
                self.assertNotEqual(result.returncode,0)
                self.assertFalse(invalid.exists())
