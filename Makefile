reinstall_package:
	@pip uninstall -y wine_advisor || :
	@pip install -e .


clean:
	@rm -fr **/__pycache__ **/*.pyc
	@rm -f **/.DS_Store
	@rm -f **/*Zone.Identifier
	@rm -f **/.ipynb_checkpoints

run_api:
	uvicorn wine_advisor.api.api_wine:app --reload
