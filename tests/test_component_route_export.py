import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from plot_component_routes import render


class RouteExportTests(unittest.TestCase):
    def test_unfinished_and_two_shortest_deliveries_and_no_overwrite(self):
        parent=ROOT/'results/thesis_test_temp';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(parent)) as directory:
            root=Path(directory);ledger=root/'ledger.json'
            data=[dict(episode=0,final_positions=[[0,0],[0,1],[1,1]],packets={'0':dict(route=[0],delay=None),'1':dict(route=[0,1,2],delay=4),'2':dict(route=[1,2],delay=2),'3':dict(route=[2],delay=1)})]
            ledger.write_text(json.dumps(data));r=render(ledger,root/'routes')
            self.assertEqual(r['unfinished'],['0']);self.assertEqual(r['delivered_examples'],['3','2'])
            for ext in ['png','svg','pdf']:self.assertGreater((root/('routes.'+ext)).stat().st_size,1000)
            with self.assertRaises(FileExistsError):render(ledger,root/'routes')


if __name__=='__main__':unittest.main()
