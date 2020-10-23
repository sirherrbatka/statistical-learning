(cl:in-package #:statistical-learning.tree-protocol)


(defun force-tree (model)
  (let ((root (root model)))
    (write-root (force-tree* (lparallel:force root))
                model)
    (setf (forced model) t)
    model))


(defun make-leaf (training-state)
  (let* ((parameters (sl.mp:training-parameters training-state))
         (result (make-leaf* parameters)))
    result))


(defun split (training-state leaf
              &optional (parameters/proxy nil proxy-p))
  (let* ((parameters (sl.mp:training-parameters training-state))
         (proxy (if proxy-p
                    parameters/proxy
                    (sl.common:proxy parameters)))
         (result (split*/proxy proxy
                               parameters
                               training-state)))
    ;; TODO also correct initialize-leaf call to include proxy
    (if (null result)
        (progn (initialize-leaf parameters training-state leaf)
               leaf)
        result)))


(defun leafs-for* (splitter node data)
  (declare (optimize (speed 3)))
  (statistical-learning.data:bind-data-matrix-dimensions ((length features-count data))
    (cl-ds:xpr (:i 0)
      (declare (type fixnum i))
      (when (< i length)
        (cl-ds:send-recur (leaf-for splitter node data i)
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
                           parallel))


(defun extract-predictions (state)
  (extract-predictions* (sl.mp:training-parameters state)
                        state))


(defun calculate-loss (state split-array)
  (calculate-loss* (training-parameters state)
                   state
                   split-array))


(defun split-training-state (state split-array
                             left-initargs right-initargs
                             &key
                               (left-size (count sl.opt:left split-array))
                               (right-size (count sl.opt:right split-array))
                             attribute-index)
  (declare (optimize (speed 3)))
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
                                   left-initargs
                                   attribute-index
                                   new-attributes)
            (split-training-state* training-parameters
                                   state
                                   split-array
                                   sl.opt:right
                                   right-size
                                   right-initargs
                                   attribute-index
                                   new-attributes))))


(defun pick-split (state)
  (let ((parameters (sl.mp:training-parameters state)))
    (pick-split* (splitter parameters)
                 parameters
                 state)))


(defun fill-split-vector (state point split-vector)
  (let ((parameters (sl.mp:training-parameters state)))
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
