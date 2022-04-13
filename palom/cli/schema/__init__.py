import pathlib
import yamale

current_dir = (pathlib.Path(__file__).resolve().parent)
svs_config_schema_path = current_dir / 'svs-config-schema.yml'
svs_config_example_path = current_dir / 'svs-config-example.yml'
cycif_config_schema_path = current_dir / 'cycif-config-schema.yml'
cycif_config_example_path = current_dir / 'cycif-config-example.yml'

svs_config_schema = yamale.make_schema(svs_config_schema_path)
cycif_config_schema = yamale.make_schema(cycif_config_schema_path)
