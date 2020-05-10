(cl:in-package #:cl-grf.tree-protocol)


(defun training-state-clone (training-state
                             new-data
                             new-target
                             new-attribute-indexes)
  (check-type training-state fundamental-training-state)
  (cl-ds.utils:quasi-clone* training-state
    :training-data new-data
    :attribute-indexes new-attribute-indexes
    :target-data new-target))


(defun force-tree (node)
  (force-tree* (lparallel:force node)))


(defun split-candidate (training-state leaf)
  (split-candidate* (training-parameters training-state)
                    training-state
                    leaf))


(defun make-leaf (training-state)
  (check-type training-state fundamental-training-state)
  (make-leaf* (training-parameters training-state)
              training-state))


(defun split (training-state leaf)
  (split* (training-parameters training-state)
          training-state
          leaf))


(defun leafs-for (node data)
  (declare (optimize (debug 3)))
  (cl-grf.data:bind-data-matrix-dimensions ((length features-count data))
    (iterate
      (with result = (make-array length))
      (for i from 0 below length)
      (setf (aref result i) (leaf-for node data i))
      (finally (return result)))))
