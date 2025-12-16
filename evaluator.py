import numpy
import struct
from collections import defaultdict
import argparse
from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """Base evaluator abstract class"""
    
    def __init__(self, dataset_path, dataset_name, train_dataset_name, 
                 test_dataset_name, algorithm, top_k, search_type, 
                 full_name, pre_index_path):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.train_dataset_name = train_dataset_name
        self.test_dataset_name = test_dataset_name
        self.algorithm = algorithm
        self.top_k = top_k
        self.search_type = search_type
        self.full_name = full_name
        self.pre_index_path = pre_index_path
    
    @staticmethod
    def read_results(filename):
        """Read binary result file"""
        result = []
        with open(filename, 'rb') as f:
            while True:
                GK_bytes = f.read(4)
                if not GK_bytes:
                    break  
                GK = struct.unpack('I', GK_bytes)[0]  
                result_bytes = f.read(4 * GK)
                result_1 = list(struct.unpack(f'{GK}I', result_bytes))
                result.append(result_1)
        return result
    
    @staticmethod
    def read_label_results(filename):
        """Read label result file"""
        result = []
        with open(filename, 'rb') as f:
            while True:
                GK_bytes = f.read(4)
                if not GK_bytes:
                    break
                GK = struct.unpack('I', GK_bytes)[0]  
                result_bytes = f.read(4 * GK)
                result_1 = list(struct.unpack(f'{GK}I', result_bytes))
                result.append(result_1)
        return result
    
    @staticmethod
    def read_query_labels(filename):
        """Read query label file"""
        labels = []
        with open(filename, 'rb') as f:
            while True:
                label_bytes = f.read(4)
                if not label_bytes:
                    break
                label = struct.unpack('I', label_bytes)[0]
                labels.append(label)
        return labels
    
    @staticmethod
    def read_groundtruth(filename):
        """Read ground truth file"""
        gt = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip().split()
                gt.append(line)
        return gt
    
    @staticmethod
    def recall(gt, res, top_k):
        """Calculate basic recall metric"""
        recall = 0
        for i in range(len(res)):
            res_i = res[i][0:top_k]
            gt_i = gt[i]
            gt_i = set([int(x) for x in gt_i])
            res_i = set(res_i)
            recall += len(gt_i & res_i)
        
        return recall / (len(res) * top_k)
    
    @abstractmethod
    def evaluate(self):
        """Execute evaluation, implemented by subclasses"""
        pass


class ImageNetEvaluator(BaseEvaluator):
    """ImageNet dataset evaluator"""
    
    def evaluate(self):
        """Execute ImageNet evaluation"""
        gt_file = f"{self.dataset_path}/result-{self.dataset_name}-{self.search_type}-top100.txt"
        res_file = f"{self.pre_index_path}/{self.algorithm}/{self.full_name}.result"
        res_label_file = f"{self.pre_index_path}/{self.algorithm}/{self.full_name}.result_top{self.top_k}.label"
        query_label_file = f"{self.dataset_path}/{self.test_dataset_name}-label.bin"
        base_label_file = f"{self.dataset_path}/{self.train_dataset_name}-label.bin"
        
        gt = self.read_groundtruth(gt_file)
        res = self.read_results(res_file)  
        res_label = self.read_label_results(res_label_file)
        gt_label = self.read_query_labels(query_label_file)
        base_label_map = self._label_mapping(base_label_file)
        
        label_acc = self._recall_label(gt_label, res_label, base_label_map, self.top_k)
        acc = self.recall(gt, res, self.top_k)
        
        print("Recall@100: ", acc)
        print("Label Recall@100: ", label_acc)
        
        for k in (5, 10, 50, 100):
            self._knn_major_classification(res_label, gt_label, k)
    
    @staticmethod
    def _label_mapping(base_label_file):
        """Build label mapping"""
        label_map = defaultdict(int)
        with open(base_label_file, 'rb') as f:
            while True:
                label_bytes = f.read(4)
                if not label_bytes:
                    break
                label = struct.unpack('I', label_bytes)[0]
                label_map[label] += 1
        return label_map
    
    @staticmethod
    def _recall_label(gt_label, res_label, base_label_map, top_k):
        """Calculate label recall"""
        recall_search = 0
        recall_base = 0
        for i in range(len(res_label)):
            res_i = res_label[i]
            gt_i = int(gt_label[i])
            recall_search += res_i.count(gt_i)
            if base_label_map[gt_i] > top_k:
                recall_base += top_k
            else:
                recall_base += base_label_map[gt_i]
        
        return recall_search / recall_base
    
    @staticmethod
    def _knn_major_classification(res_label, gt_label, top_k):
        """K-nearest neighbors majority classification"""
        recall = 0
        for i in range(len(res_label)):
            res_i = res_label[i]
            gt_i = int(gt_label[i])
            res_i = res_i[:top_k]
            res_i = [int(x) for x in res_i]
            res_i = numpy.array(res_i)
            res_i = numpy.bincount(res_i)
            res_i = numpy.argmax(res_i)
            if res_i == gt_i:
                recall += 1
        
        print(f"Majority Classification Recall@{top_k}:\t", recall / len(res_label))


class BookEvaluator(BaseEvaluator):
    """Book dataset evaluator"""
    
    def evaluate(self):
        """Execute Book evaluation"""
        gt_file = f"{self.dataset_path}/result-{self.dataset_name}-{self.search_type}-top100.txt"
        res_file = f"{self.pre_index_path}/{self.algorithm}/{self.full_name}.result"
        res_label_file = f"{self.pre_index_path}/{self.algorithm}/{self.full_name}.result_top{self.top_k}.label"
        
        gt = self.read_groundtruth(gt_file)
        res = self.read_results(res_file)
        res_label = self.read_query_labels(res_label_file)
        
        acc = self.recall(gt, res, self.top_k)
        acc_self = self._recall_label(res, res_label, self.top_k)
        
        print("Recall@100: ", acc)
        print("Recall_self@100: ", acc_self)
    
    @staticmethod
    def _recall_label(res, res_label, top_k):
        """Calculate self-label recall"""
        recall = 0
        for i in range(len(res)):
            res_i = res[i][0:top_k]
            gt_label = int(res_label[i])
            if gt_label in res_i:
                recall += 1
        return recall / len(res)


class CommerceEvaluator(BaseEvaluator):
    """Commerce dataset evaluator"""
    
    def evaluate(self):
        """Execute Commerce evaluation"""
        query_type = 'high'
        
        gt_file = f"{self.dataset_path}/result-{self.dataset_name}-{query_type}-{self.search_type}-top100.txt"
        res_file = f"{self.pre_index_path}/{self.algorithm}/{self.full_name}.result"
        query_label_file = f"{self.dataset_path}/{self.test_dataset_name}_label.bin"
        base_label_file = f"{self.dataset_path}/{self.train_dataset_name}_label.bin"
        meta_file = f"{self.dataset_path}/q_{query_type}_meta.txt"
        
        gt = self.read_groundtruth(gt_file)
        res = self.read_results(res_file)
        query_label = self.read_query_labels(query_label_file)
        base_label = self.read_query_labels(base_label_file)
        meta_query_data = self._extract_query_data(meta_file)
        
        acc = self.recall(gt, res, self.top_k)
        pop_score, _ = self._recall_hot(meta_query_data, query_label, res, base_label, self.top_k, 1)
        
        print("Recall@100: ", acc)
        print(f"Popularity Score@100: ", pop_score)
    
    @staticmethod
    def _parse_user_interact_items(items_str):
        """Parse user interaction items"""
        items = []
        for item in items_str.split(';'):
            if item:
                item_id, label = item.split(':')
                items.append((int(item_id), int(label)))
        return items
    
    @staticmethod
    def _parse_hot_items(hot_items_str):
        """Parse hot items"""
        hot_items = []
        for item in hot_items_str.split(';'):
            if item:
                item_id, hot_levels = item.split(':')
                hot_level1, hot_level2 = map(float, hot_levels.split(','))
                hot_items.append((int(item_id), hot_level1, hot_level2))
        return hot_items
    
    @staticmethod
    def _extract_query_data(file_path):
        """Extract query data"""
        query_data = {}
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                q_id, q_type, user_interact_items, hot_items = line.split('|')
                
                q_id = int(q_id)
                q_type = int(q_type)
                user_interact_items = CommerceEvaluator._parse_user_interact_items(user_interact_items)
                hot_items = CommerceEvaluator._parse_hot_items(hot_items)
                
                query_data[q_id] = {
                    'q_type': q_type,
                    'user_interact_items': user_interact_items,
                    'hot_items': hot_items
                }
        
        return query_data
    
    @staticmethod
    def _recall_hot(query_data, query_label, res, base_label, top_k, hot_label_type):
        """Calculate hot recall"""
        recall_hot = 0
        hot_tot = 0
        
        for i in range(len(res)):
            res_i = res[i][:top_k]
            res_i = [int(base_label[x]) for x in res_i]
            hot_items = query_data[query_label[i]]['hot_items']
            hot_items_set = {item_id for item_id, _, _ in hot_items}
            hot_item_hot_level = {item_id: (hot_level1, hot_level2) for item_id, hot_level1, hot_level2 in hot_items}
            hot_tot += sum(hot_item_hot_level[item_id][hot_label_type] for item_id in hot_items_set)
            recall_hot += sum(hot_item_hot_level[item_id][hot_label_type] for item_id in res_i if item_id in hot_items_set)
        
        return recall_hot, recall_hot / hot_tot if hot_tot != 0 else 0


class GlinkEvaluator(BaseEvaluator):
    """Glink dataset evaluator"""
    
    def evaluate(self):
        """Execute Glink evaluation"""
        gt_file = f"{self.dataset_path}/result-{self.dataset_name}-{self.search_type}-top100.txt"
        res_file = f"{self.pre_index_path}/{self.algorithm}/{self.full_name}.result"
        res_label_file = f"{self.pre_index_path}/{self.algorithm}/{self.full_name}.result_top{self.top_k}.label"
        query_label_file = f"{self.dataset_path}/{self.test_dataset_name}-label.bin"
        base_label_file = f"{self.dataset_path}/{self.train_dataset_name}-label.bin"
        
        gt = self.read_groundtruth(gt_file)
        res = self.read_results(res_file)  
        res_label = self.read_label_results(res_label_file)
        gt_label = self.read_query_labels(query_label_file)
        base_label_map = self._label_mapping(base_label_file)
        
        label_acc = self._recall_label(gt_label, res_label, base_label_map, 100)
        acc = self.recall(gt, res, self.top_k)
        
        print("Recall@100: ", acc)
        print("Label Recall@100: ", label_acc)
    
    @staticmethod
    def _label_mapping(base_label_file):
        """Build label mapping"""
        label_map = defaultdict(int)
        with open(base_label_file, 'rb') as f:
            while True:
                label_bytes = f.read(4)
                if not label_bytes:
                    break
                label = struct.unpack('I', label_bytes)[0]
                label_map[label] += 1
        return label_map
    
    @staticmethod
    def _recall_label(gt_label, res_label, base_label_map, top_k):
        """Calculate label recall"""
        recall_search = 0
        recall_base = 0
        for i in range(len(res_label)):
            res_i = res_label[i]
            gt_i = int(gt_label[i])
            recall_search += res_i.count(gt_i)
            if base_label_map[gt_i] > top_k:
                recall_base += top_k
            else:
                recall_base += base_label_map[gt_i]
        
        return recall_search / recall_base


class EvaluatorFactory:
    """Evaluator factory class"""
    
    _evaluators = {
        'imagenet': ImageNetEvaluator,
        'book': BookEvaluator,
        'commerce': CommerceEvaluator,
        'glink': GlinkEvaluator,
    }
    
    @classmethod
    def create_evaluator(cls, dataset_type, *args, **kwargs):
        """Create evaluator of the specified type"""
        if dataset_type not in cls._evaluators:
            raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                           f"Supported types: {', '.join(cls._evaluators.keys())}")
        
        evaluator_class = cls._evaluators[dataset_type]
        return evaluator_class(*args, **kwargs)
    
    @classmethod
    def register_evaluator(cls, dataset_type, evaluator_class):
        """Register a new evaluator type"""
        cls._evaluators[dataset_type] = evaluator_class


def main():
    parser = argparse.ArgumentParser(description="Calculate recall for a given dataset and algorithm.")
    parser.add_argument("DATA_PRE_PATH", type=str, help="Path to the dataset")
    parser.add_argument("PREFIX", type=str, help="Dataset name")
    parser.add_argument("train_dataset_name", type=str, help="Train dataset name")
    parser.add_argument("test_dataset_name", type=str, help="Test dataset name")
    parser.add_argument("algorithm", type=str, help="Algorithm name")
    parser.add_argument("top_k", type=int, help="Top K value")
    parser.add_argument("type", type=str, help="Search type")
    parser.add_argument("full_name", type=str, help="Full name")
    parser.add_argument("pre_path", type=str, help="Index prefix path")
    parser.add_argument("--dataset-type", type=str, default="imagenet",
                       choices=['imagenet', 'book', 'commerce', 'glink'],
                       help="Dataset type (default: imagenet)")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = EvaluatorFactory.create_evaluator(
        args.dataset_type,
        args.DATA_PRE_PATH,
        args.PREFIX,
        args.train_dataset_name,
        args.test_dataset_name,
        args.algorithm,
        args.top_k,
        args.type,
        args.full_name,
        args.pre_path
    )
    
    # Execute evaluation
    evaluator.evaluate()


if __name__ == "__main__":
    main()
