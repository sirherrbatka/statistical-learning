(cl:in-package #:statistical-learning.club-drf)


(defmethod sl.ensemble:prune-trees ((algorithm club-drf)
                                    ensemble
                                    train-data
                                    target-data)
  (declare (ignore target-data))
  (let* ((sample-size (sample-size algorithm))
         (tree-labels (~>> (sl.data:draw-random-data-points-subset sample-size
                                                                   train-data)
                           first
                           (obtain-labels algorithm ensemble)))
         (parallel (parallel algorithm))
         (trees-count (number-of-trees-selected algorithm))
         (max-neighbor (max-neighbor algorithm))
         (clustering-parameters (make 'clusters.clarans:parameters
                                      :parallelp parallel
                                      :medoids-count trees-count
                                      :max-neighbor max-neighbor
                                      :distance-function #'diversity-measure))
         (clusters (clusters:cluster clustering-parameters
                                     tree-labels))
         (cluster-contents (clusters:cluster-contents clusters)))
    (cl-ds.utils:quasi-clone ensemble
                             :trees (map 'vector
                                         (compose #'car #'first-elt)
                                         (remove-if #'emptyp cluster-contents)))))
