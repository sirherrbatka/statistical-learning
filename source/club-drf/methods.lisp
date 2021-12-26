(cl:in-package #:statistical-learning.club-drf)


(defmethod sl.ensemble:prune-trees ((algorithm club-drf)
                                    ensemble
                                    train-data
                                    target-data)
  (bind ((sample-size (sample-size algorithm))
         ((train-data target-data)
          (sl.data:draw-random-data-points-subset sample-size
                                                  train-data
                                                  target-data))
         (tree-labels (obtain-labels algorithm ensemble train-data))
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
         (cluster-contents (clusters:cluster-contents clusters))
         (use-cluster-size (use-cluster-size algorithm))
         (use-accuracy (use-accuracy algorithm))
         ((:flet impl (cluster &aux
                               (medoid (~> cluster first-elt car))
                               (labels (~> cluster first-elt cdr))
                               (size (length cluster))))
          (cl-ds.utils:quasi-clone* medoid
            :weight (* (if use-cluster-size
                           (coerce size 'single-float)
                           1.0f0)
                       (if use-accuracy
                           (iterate
                             (with attributes-count = (sl.data:attributes-count target-data))
                             (for i from 0 below (sl.data:data-points-count target-data))
                             (for target = (sl.data:mref target-data i 0))
                             (counting (= (iterate
                                            (for j from 0 below attributes-count)
                                            (finding j maximizing (sl.data:mref labels i j)))
                                          target)))
                           1.0f0)))))
    (cl-ds.utils:quasi-clone ensemble
                             :trees (map 'vector
                                         #'impl
                                         (remove-if #'emptyp cluster-contents)))))
