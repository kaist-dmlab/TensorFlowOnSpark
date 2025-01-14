class DBParser():
	def __init__(self):
		print("==========Start DBParser ==========")
		self.query = None
  
	def read_query(self, query_path):
		file = open(query_path, "r")
		query = file.read()   
		print("read query:\n", query)
		file.close()
		return query

	def split_query(self, query_path):
		query = self.read_query(query_path)
		return query.strip().replace("\n", " ").split()      

	def get_index(self, statement):
		return self.query.index(statement) if statement in self.query else None

	def parse(self, sql_query_path):
     	
		self.query = self.split_query(sql_query_path)

		# Dealing INSERT Query
		if self.query[0] == "INSERT":
			in_local_path  = self.query[self.get_index("INSERT")+1]
			out_hdfs_path  = self.query[self.get_index("INTO")+1]
			num_partitions = self.query[self.get_index("PARTITIONS")+1]

			return in_local_path, out_hdfs_path, num_partitions
			#self.db_manager.insert(in_local_path, out_hdfs_path, num_partitions)

		# Dealing SELECT Query
		if self.query[0] == "SELECT":
			sql_query = " ".join(self.query[:self.get_index("FOR")])
			sql_query = " ".join(sql_query.split()[:3] + ['temp'] + sql_query.split()[4:])
			hdfs_path  = self.query[self.get_index("FROM")+1]
			task, task_path  = self.query[self.get_index("FOR")+1:]

			return sql_query, hdfs_path, task, task_path
			# self.db_manager.select(sql_query, hdfs_path, task, task_path)
