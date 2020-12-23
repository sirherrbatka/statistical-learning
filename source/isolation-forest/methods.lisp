(cl:in-package #:statistical-learning.isolation-forest)


(defmethod sl.tp:split*/proxy
    (parameters/proxy
     (training-parameters isolation)
     training-state)
  (bind ((split-array (~> training-state sl.mp:data-points length
                          sl.opt:make-split-array))
         (point (sl.tp:pick-split training-state)))
    (bind (((:values left-length right-length)
            (sl.tp:fill-split-vector training-state
                                     split-array))
           (new-depth (~> training-state sl.tp:depth 1+))
           ((:flet new-state (position size))
            (sl.tp:split-training-state*
             training-parameters
             training-state
             split-array
             position
             size
             `(:depth ,new-depth)
             point))
           ((:flet subtree-impl
              (position size &aux (state (new-state position size))))
            (~>> (sl.tp:make-leaf* training-parameters)
                 (sl.tp:split state)))
           ((:flet subtree (position size &optional parallel))
            (if (and parallel (< new-depth 10))
                (lparallel:future (subtree-impl position size))
                (subtree-impl position size)))
           (parallel (sl.tp:parallel training-parameters)))
      (sl.tp:make-node 'sl.tp:fundamental-tree-node
                       :left-node (subtree sl.opt:left
                                           left-length
                                           parallel)
                       :right-node (subtree sl.opt:right
                                            right-length)
                       :point point))))


(defmethod sl.tp:leaf-for/proxy (splitter/proxy
                                 (splitter isolation-splitter)
                                 node
                                 data
                                 index
                                 context)
  (declare (type sl.data:double-float-data-matrix data)
           (type fixnum index))
  (bind ((normals (normals splitter))
         (global-min (global-min context))
         (global-max (global-max context))
         (mins (mins context))
         (maxs (maxs context))
         ((:labels impl
            (node depth &aux (next-depth (the fixnum (1+ depth)))))
          (declare (optimize (speed 3) (safety 0)))
          (if (sl.tp:treep node)
              (if (rightp (sl.tp:point node)
                          normals index data
                          mins maxs
                          global-min global-max)
                  (~> node sl.tp:right-node (impl next-depth))
                  (~> node sl.tp:left-node (impl next-depth)))
              (values node depth))))
    (impl node 0)))


(defmethod sl.tp:requires-split-p/proxy and
    (parameters/proxy
     (splitter isolation-splitter)
     (isolation isolation)
     training-state)
  (and (> (~> training-state sl.mp:data-points length (* 2))
          (sl.tp:minimal-size isolation))
       (< (sl.tp:depth training-state)
          (sl.tp:maximal-depth isolation))))


(defmethod sl.mp:make-training-state/proxy (parameters/proxy
                                            (parameters isolation)
                                            &rest initargs
                                            &key
                                              train-data
                                              mins
                                              maxs
                                              global-min
                                              global-max
                                              data-points
                                              attributes)
  (declare (ignore initargs))
  (make-instance 'isolation-training-state
                 :depth 0
                 :mins mins
                 :maxs maxs
                 :global-min global-min
                 :global-max global-max
                 :parameters parameters
                 :train-data train-data
                 :data-points data-points
                 :attributes attributes))


(defmethod sl.tp:make-leaf*/proxy (training-parameters/proxy
                                   (training-parameters isolation))
  (make 'isolation-leaf))


(defmethod sl.tp:initialize-leaf/proxy (training-parameters/proxy
                                        (training-parameters isolation)
                                        training-state
                                        leaf)
  leaf)


(defmethod sl.tp:pick-split*/proxy (splitter/proxy
                                    (splitter isolation-splitter)
                                    parameters
                                    state)
  (bind ((data-points (sl.mp:data-points state))
         (data (sl.mp:train-data state))
         (depth (sl.tp:depth state))
         (normals (normals splitter))
         (maximal-depth (sl.tp:maximal-depth parameters))
         (depth-ratio (- 1.0d0 (/ depth maximal-depth)))
         (attributes (sl.tp:attribute-indexes state))
         (mins (mins state))
         (maxs (maxs state))
         (global-min (global-min state))
         (global-max (global-max state))
         (gaussian-state (gaussian-state state)))
    (make-isolation-forest-split-point
     :dot-product (wdot (generate-point data
                                        data-points
                                        attributes
                                        depth-ratio
                                        (mins state)
                                        (maxs state)
                                        gaussian-state)
                       normals
                       0
                       0
                       attributes
                       mins
                       maxs
                       global-min
                       global-max)
     :attributes attributes)))


(defmethod fill-split-vector*/proxy
    (splitter/proxy
     (splitter isolation-splitter)
     parameters
     state
     point
     split-vector)
  (declare (type sl.data:split-vector split-vector)
           (type isolation-forest-split-point point)
           (optimize (speed 0) (safety 3) (debug 3)))
  (bind ((data (sl.mp:train-data state))
         (data-points (sl.mp:data-points state))
         (mins (mins state))
         (maxs (maxs state))
         (global-min (global-min state))
         (global-max (global-max state))
         (normals (~> parameters
                      sl.tp:splitter
                      normals)))
    (declare (type sl.data:double-float-data-matrix normals)
             (type (simple-array fixnum (*)) data-points))
    (iterate
      (declare (type fixnum right-count left-count i j))
      (with right-count = 0)
      (with left-count = 0)
      (for j from 0 below (length data-points))
      (for i = (aref data-points j))
      (for rightp = (rightp point normals i data
                            mins maxs global-min global-max))
      (setf (aref split-vector j) rightp)
      (if rightp (incf right-count) (incf left-count))
      (finally (return (values left-count right-count))))))


(defmethod cl-ds.utils:cloning-information append
    ((object isolation-training-state))
  '((:parameters sl.mp:training-parameters)
    (:depth sl.tp:depth)
    (:averages averages)
    (:mins mins)
    (:maxs maxs)
    (:global-min global-min)
    (:global-max global-max)
    (:train-data sl.mp:train-data)
    (:attributes sl.tp:attribute-indexes)
    (:data-points sl.mp:data-points)
    (:gaussian-state gaussian-state)))


(defmethod sl.tp:extract-predictions*/proxy (parameters/proxy
                                             (parameters isolation)
                                             state)
  (let* ((trees-count (trees-count state))
         (trees-sum (trees-sum state))
         (c (c state))
         (result (sl.data:make-data-matrix-like trees-sum)))
    (iterate
      (for i from 0 below (sl.data:data-points-count trees-sum))
      (for avg = (/ (sl.data:mref trees-sum i 0)
                    trees-count))
      (for value = (/ 1 (expt 2 (/ avg c))))
      (setf (sl.data:mref result i 0) value)
      (finally (return result)))))


(defmethod sl.tp:contribute-predictions*/proxy
    (parameters/proxy
     (parameters isolation)
     tree-model
     data
     state
     context
     parallel
     &optional (leaf-key #'identity))
    (when (null state)
      (let ((c (c context))
            (data-points-count (sl.data:data-points-count data)))
        (check-type c double-float)
        (setf state (make 'isolation-prediction
                          :indexes data-points-count
                          :trees-sum (sl.data:make-data-matrix
                                      data-points-count
                                      1)
                          :c (c context)))))
  (let* ((splitter (sl.tp:splitter parameters))
         (predictions-lock (sl.tp:predictions-lock state))
         (sums (trees-sum state))
         (root (sl.tp:root tree-model)))
    (funcall (if parallel #'lparallel:pmap #'map)
             nil
             (lambda (data-point)
               (bind (((:values leaf depth)
                       (~>> (sl.tp:leaf-for splitter root
                                            data data-point
                                            context)
                            (funcall leaf-key))))
                 (declare (ignore leaf))
                 (bt:with-lock-held (predictions-lock)
                   (incf (sl.data:mref sums data-point 0)
                         depth))))
             (sl.tp:indexes state))
    (incf (trees-count state))))
