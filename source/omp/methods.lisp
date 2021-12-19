(cl:in-package #:statistical-learning.omp)


(defmethod prune-trees ((parameters sl.perf:regression)
                        omp
                        ensemble
                        trees
                        train-data
                        target-data)
  (bind ((predictions (extract-predictions ensemble
                                           trees
                                           train-data
                                           nil))
         (dictionary (extract-predictions-column predictions
                                                 0))
         (selected-trees (omp target-data
                              dictionary
                              (number-of-trees-selected omp))))
    (map 'vector (curry #'aref trees) selected-trees)))


(defmethod sl.ensemble:prune-trees ((algorithm orthogonal-matching-pursuit)
                                    (ensemble-model sl.ensemble:ensemble-model)
                                    train-data
                                    target-data)
  (bind ((ensemble (sl.mp:parameters ensemble-model))
         (parameters (sl.ensemble:tree-parameters ensemble)))
    (cl-ds.utils:quasi-clone
     ensemble-model
     :trees (prune-trees parameters
                         algorithm
                         ensemble
                         (sl.ensemble:trees ensemble-model)
                         train-data
                         target-data))))
