(cl:in-package #:statistical-learning.tree-protocol)


(defun force-tree (model)
  (let ((root (root model)))
    (write-root (force-tree* (lparallel:force root))
                model)
    (setf (forced model) t)
    model))


(defun split-candidate (training-state leaf)
  (split-candidate* (training-parameters training-state)
                    training-state
                    leaf))


(defun make-leaf (training-state)
  (check-type training-state tree-training-state)
  (let* ((parameters (sl.mp:training-parameters training-state))
         (result (make-leaf* parameters
                             training-state)))
    (initialize-leaf parameters training-state result)
    result))


(defun split (training-state leaf)
  (let ((result (split* (sl.mp:training-parameters training-state)
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
  (contribute-predictions* (sl.mp:training-parameters model)
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


(defun leaf-for (node data index)
  (declare (type statistical-learning.data:data-matrix data)
           (type fixnum index))
  (if (typep node 'fundamental-leaf-node)
      node
      (bind ((attribute-index (attribute node))
             (attribute-value (attribute-value node)))
        (if (> (statistical-learning.data:mref data index attribute-index)
               attribute-value)
            (leaf-for (right-node node) data index)
            (leaf-for (left-node node) data index)))))


(defun split-training-state (state split-array
                             left-arguments right-arguments
                             &key
                               (left-size (count sl.opt:left split-array))
                               (right-size (count sl.opt:right split-array))
                             attribute-index)
  (let* ((training-parameters (sl.mp:training-parameters state))
         (attribute-indexes (attribute-indexes state))
         (new-attributes (if (null attribute-index)
                             attribute-indexes
                             (subsample-vector attribute-indexes
                                               attribute-index))))
    (values (split-training-state* training-parameters
                                   state
                                   split-array
                                   sl.opt:left
                                   left-size
                                   left-arguments
                                   attribute-index
                                   new-attributes)
            (split-training-state* training-parameters
                                   state
                                   split-array
                                   sl.opt:right
                                   right-size
                                   right-arguments
                                   attribute-index
                                   new-attributes))))
