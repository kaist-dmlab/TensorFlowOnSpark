class DBParser():
    def __init__(self):
        print("----------Start DBParser ----------")
        self.db_manager = DBManager()
    
    
    def make_query_split(self, query):
        """
        query       = INSERT mnist, cifar10 INTO hdfs/dataset/ 
        query_split = ['INSERT', 'mnist', 'cifar10', 'INTO', 'hdfs/dataset/']
        """
        query = query.strip()
        query = query.replace("\n", " ").replace("\t", " ").replace(",", " ")      
        query_split = query.split()

        return query_split

    def parse(self, query):
        
        query_split = self.make_query_split(query)
        
        # Dealing INSERT Query
        if query_split[0].upper() == "INSERT":
            
            for idx, query_segment in enumerate(query_split):
                if query_segment.upper() == "INSERT": INSERT_idx = idx
                if query_segment.upper() == "INTO"  : INTO_idx   = idx
                       
            in_local_path_list = query_split[INSERT_idx+1:INTO_idx]
            out_hdfs_path = query_split[INTO_idx+1]
            
            # can insert more than 2 datasets
            for in_local_path in in_local_path_list:
                self.db_manager.insert(in_local_path, out_hdfs_path)
        
        
        # Dealing SELECT Query
        if query_split[0].upper() == "SELECT":
            
            SELECT_idx, FROM_idx, WHERE_idx, FOR_idx = None, None, None, None
            
            for idx, query_segment in enumerate(query_split):
                if query_segment.upper() == "SELECT": SELECT_idx = idx
                if query_segment.upper() == "FROM"  : FROM_idx   = idx
                if query_segment.upper() == "WHERE" : WHERE_idx  = idx
                if query_segment.upper() == "FOR"   : FOR_idx    = idx
             
            col_names = query_split[SELECT_idx+1:FROM_idx]
            filepath = query_split[FROM_idx+1]
            task = query_split[FOR_idx+1]
            
            if WHERE_idx == None:
                self.db_manager.select(col_names = col_names, filepath = filepath, task = task)
            else:
                sampler = query_split[WHERE_idx+1]
                sampling_rate = query_split[WHERE_idx+2]
                self.db_manager.select(col_names, filepath, sampler, sampling_rate, task)