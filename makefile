clean:
	rm -r __pycache__
	rm -r */__pycache__

clean_log:
	rm -r logs

clean_save:
	rm -rf modeling/save/*

clean_result:
	rm result/*

clean_vector:
	rm -r modeling/vectors/vectors_*
