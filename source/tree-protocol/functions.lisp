(cl:in-package #:statistical-learning.tree-protocol)


(defun force-tree (model)
  (let ((root (root model)))
    (write-root (force-tree* (lparallel:force root))
                model)
    (setf (forced model) t)
    model))


(defun make-leaf (training-state)
  (let* ((parameters (sl.mp:training-parameters training-state))
         (result (make-leaf* parameters training-state)))
    (assert (not (null result)))
    (initialize-leaf parameters training-state result)
    result))


(defun leafs-for* (splitter node data)
  (declare (optimize (speed 3)))
  (statistical-learning.data:bind-data-matrix-dimensions ((length features-count data))
    (cl-ds:xpr (:i 0)
      (declare (type fixnum i))
      (when (< i length)
        (cl-ds:send-recur (leaf-for splitter node data i nil)
                          :i (1+ i))))))


(defun leafs-for (model data)
  (leafs-for* (~> model sl.mp:training-parameters splitter)
              (root model)
              data))


(defun visit-nodes* (tree-node function
                    &key (filter-function (constantly t)))
  (check-type tree-node fundamental-node)
  (labels ((impl (node &optional parent)
             (when (funcall filter-function node)
               (funcall function node parent))
             (when (treep node)
               (impl (left-node node) node)
               (impl (right-node node) node))))
    (impl tree-node)
    tree-node))


(defun visit-nodes (model function &key (filter-function (constantly t)))
  (visit-nodes* (root model) function :filter-function filter-function))


(defun contribute-predictions (model data state parallel)
  (contribute-predictions* (sl.mp:training-parameters model)
                           model
                           data
                           state
                           nil
                           parallel))


(defun extract-predictions (state)
  (extract-predictions* (sl.mp:training-parameters state)
                        state))


(defun split-training-state (state split-array
                             position
                             size
                             initargs
                             point)
  (declare (optimize (speed 3)))
  (split-training-state* (sl.mp:training-parameters state)
                         state
                         split-array
                         position
                         size
                         initargs
                         point))


(defun pick-split (state)
  (let ((parameters (sl.mp:training-parameters state)))
    (pick-split* (splitter parameters)
                 parameters
                 state)))


(defun fill-split-vector (state split-vector)
  (let ((parameters (sl.mp:training-parameters state))
        (point (split-point state)))
    (fill-split-vector* (splitter parameters)
                        parameters
                        state
                        point
                        split-vector)))


(defun random-attribute-splitter ()
  (make 'random-attribute-splitter))


(defun distance-splitter (distance-function &optional (iterations 0) (repeats 8))
  (ensure-functionf distance-function)
  (make 'distance-splitter
        :iterations iterations
        :repeats repeats
        :distance-function distance-function))


(defun split (training-state)
  (let* ((parameters (sl.mp:training-parameters training-state))
         (splitter (splitter parameters)))
    (or (when (requires-split-p splitter parameters training-state)
          (split* (sl.mp:training-parameters training-state)
                  training-state))
        (make-leaf training-state))))


(defun make-tree (training-state)
  (split training-state))


(defun gather-split-points (state)
  (iterate
    (for s initially state then (parent-state s))
    (while s)
    (collecting (split-point s))))
