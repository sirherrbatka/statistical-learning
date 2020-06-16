(cl:in-package #:statistical-learning.tree-protocol)


(defun force-tree (model)
  (let ((root (root model)))
    (write-root (force-tree* (lparallel:force root))
                model)
    model))


(defun split-candidate (training-state leaf)
  (split-candidate* (training-parameters training-state)
                    training-state
                    leaf))


(defun make-leaf (training-state)
  (check-type training-state fundamental-training-state)
  (make-leaf* (training-parameters training-state)
              training-state))


(defun split (training-state leaf)
  (let ((result (split* (training-parameters training-state)
                        training-state
                        leaf)))
    (if (null result)
        leaf
        result)))


(defun leafs-for* (node data)
  (declare (optimize (speed 3)))
  (statistical-learning.data:bind-data-matrix-dimensions ((length features-count data))
    (cl-ds:xpr (:i 0)
      (declare (type fixnum i))
      (when (< i length)
        (cl-ds:send-recur (leaf-for node data i)
                          :i (1+ i))))))


(defun leafs-for (model data)
  (leafs-for* (root model) data))


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
  (contribute-predictions* (training-parameters model)
                           model
                           data
                           state
                           parallel))


(defun extract-predictions (state)
  (extract-predictions* (training-parameters state)
                        state))


(defun calculate-loss (state split-array)
  (calculate-loss* (training-parameters state)
                   state
                   split-array))
