clean:
	rm -r __pycache__
	rm -r */__pycache__

clean-log:
	rm -r logs/*

clean-save:
	rm -rf modeling/save/*

clean-result:
	rm result/*

clean-vector:
	rm -r modeling/vectors/vectors_*
