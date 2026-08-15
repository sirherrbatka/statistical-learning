(cl:in-package #:statistical-learning.htm)


(defun burst-column (neurons result-hash-table)
  "Activate all neurons in a column to signal a prediction error.

When spatial pooling selects a leaf but no neuron within it exceeded the
prediction threshold, bursting activates every cell in that column by setting
them to T in RESULT-HASH-TABLE. This signals surprise and triggers learning
for unexpected input contexts."
  (map nil
       (lambda (neuron)
         (setf (gethash neuron result-hash-table) t))
       neurons))

(defun sum-distal-weights (activity-hash-table neuron-weight-table)
  "Accumulate distal weights for a set of target neurons.

Iterates over NEURON-WEIGHT-TABLE (Target -> Weight) and adds each weight to
the corresponding neuron's running total in ACTIVITY-HASH-TABLE, initializing
missing entries to 0.0 before incrementing."
  (maphash (lambda (neuron weight)
             (incf (gethash neuron activity-hash-table 0.0) weight))
           neuron-weight-table))

(defun propagate-one-source-distal-activity (leaf-weights activity-hash-table)
  "Propagate distal activation from a single previously-active source neuron.

Iterates over all leaves connected to this source (via LEAF-WEIGHTS, which maps
Leaf -> Target -> Weight) and accumulates synapse weights into
ACTIVITY-HASH-TABLE for every reachable target neuron across spatial contexts."
  (maphash-values (lambda (neuron-weights-leafs-weights)
                    (sum-distal-weights activity-hash-table
                                        neuron-weights-leafs-weights))
                  leaf-weights))

(defun compute-distal-activation (weights previously-active-neurons activity-hash-table)
  "Propagate temporal context from previously-active source neurons to their targets.

Iterates over PREVIOUSLY-ACTIVE-NEURONS and, for each source that has entries
in WEIGHTS (stored as Source -> Leaf -> Target -> Weight), accumulates
distal activation into ACTIVITY-HASH-TABLE by summing incoming synapse
weights across all connected leaves and target neurons.

This phase runs before column resolution: it populates the activity table
for the entire network, not just spatially active regions. Column-level
filtering against ACTIVE-LEAFS happens later in RESOLVE-COLUMN-ACTIVATION."
  (maphash-keys (lambda (neuron)
                  (when-let ((column-table (gethash neuron weights)))
                    (propagate-one-source-distal-activity column-table
                                                          activity-hash-table)))
                previously-active-neurons))

(defun resolve-column-activation (active-leafs activity-hash-table currently-active-neurons parameters)
  "Resolve column activation based on distal prediction activity.

For each spatially active leaf, checks whether any of its neurons have
accumulated distal activity (in ACTIVITY-HASH-TABLE) exceeding the
ACTIVATION-THRESHOLD from PARAMETERS. If at least one neuron exceeds the
threshold, those predicted neurons are activated by setting them in
CURRENTLY-ACTIVE-NEURONS. If no neuron meets the threshold, the entire
column bursts via BURST-COLUMN to signal a prediction error (surprise).

Returns the number of leaves that burst during this time step."
  (let ((bursting-count 0)
        (threshold (activation-threshold parameters)))
    (loop for leaf being the hash-keys of active-leafs do
      (let* ((neurons (column-neurons leaf))
             (predicted-p nil))
        (declare (type simple-vector neurons))
        (iterate
          (for n in-vector neurons)
          (when (> (gethash n activity-hash-table 0.0) threshold)
            (setf predicted-p t
                  (gethash n currently-active-neurons) t)))
        (unless predicted-p
          (incf bursting-count)
          (burst-column neurons currently-active-neurons))))
    bursting-count))

(defun decay-false-positive-weights (weights previously-active-neurons currently-active-neurons parameters)
  "Decay distal weights for synapses from previously-active sources whose targets
did not fire this step (false positives).

Since weights are stored as Source -> Leaf -> Target, we iterate over
previously-active source neurons and, within each, every leaf they connect to.
A target that is in CURRENTLY-ACTIVE-NEURONS is a true positive and is left
untouched; every other target -- whether its leaf was spatially inactive or it
simply lost intra-column competition -- is a false positive: its weight is shrunk
by DECAY-RATE, and synapses falling below DROP-OUT-THRESHOLD are pruned via remhash.

After each pass, leaf tables left empty by pruning are removed from their source
table, and source tables left empty are removed from WEIGHTS, so the weight
structure does not grow unboundedly over training. Deleting keys while maphash-ing
a table is spec-compliant (CLHS 3.2.2.5)."
  (let* ((decay-rate (decay-rate parameters))
         (multiplier (- 1.0 decay-rate))
         (drop-out-threshold (drop-out-threshold parameters)))
    (maphash-keys (lambda (source-neuron)
                    (when-let ((source-table (gethash source-neuron weights)))
                      ;; Iterate over all leaves this source connects to
                      (maphash (lambda (leaf leaf-targets)
                                 ;; Iterate over all targets in this leaf
                                 (maphash (lambda (target-neuron weight)
                                            (unless (gethash target-neuron currently-active-neurons)
                                              (let ((new-weight (* weight multiplier)))
                                                (if (< new-weight drop-out-threshold)
                                                    (remhash target-neuron leaf-targets)
                                                    (setf (gethash target-neuron leaf-targets) new-weight)))))
                                          leaf-targets)
                                 ;; Reclaim the leaf table if pruning emptied it
                                 (when (zerop (hash-table-count leaf-targets))
                                   (remhash leaf source-table)))
                               source-table)
                      ;; Reclaim the source table if all its leaves were dropped
                      (when (zerop (hash-table-count source-table))
                        (remhash source-neuron weights))))
                  previously-active-neurons)))

(defun learn-distal-weights (leaf-weights active-neurons column-neurons learning-rate)
  "Update distal weights for one (source, leaf) context. Only targets that are
both in COLUMN-NEURONS and currently in ACTIVE-NEURONS are reinforced
(true positives) or newly formed (false negatives), keeping each synapse
scoped to its own spatial context."
  (iterate
    (for target in-vector column-neurons)
    (when (gethash target active-neurons)
      (if-let ((old-weight (gethash target leaf-weights)))
        (setf (gethash target leaf-weights)
              (min 1.0 (+ old-weight (* learning-rate (- 1.0 old-weight)))))
        (setf (gethash target leaf-weights) learning-rate)))))

(defun reinforce-active-synapses (previously-active-neurons active-neurons active-leafs weights parameters)
  "Update distal synapses for all neurons in PREVIOUSLY-ACTIVE-NEURONS.

For each previously-active source neuron, lazily initializes the intermediate 
hash-tables mapping source -> leaf (skipping inactive leaves) -> target, 
and then delegates to LEARN-DISTAL-WEIGHTS to reinforce or form connections."
  (let ((learning-rate (learning-rate parameters)))
    (maphash-keys (lambda (source-neuron)
                    ;; Level 1: Ensure target-level table exists
                    (let ((neuron-weights (ensure (gethash source-neuron weights)
                                              (make-hash-table :test 'eq))))
                      ;; Level 2: Create leaf-level tables only for spatially active
                      (maphash-keys (lambda (leaf)
                                      (let ((leaf-weights (ensure (gethash leaf neuron-weights)
                                                            (make-hash-table :test 'eq))))
                                        ;; Level 3: learn-distal-weights handles source-level lazy init
                                        (learn-distal-weights leaf-weights
                                                              active-neurons
                                                              (column-neurons leaf)
                                                              learning-rate)))
                                    active-leafs)))
                  previously-active-neurons)))

(defun make-neuron ()
  (make-instance 'neuron))

(defun ensure-column-initialization (column cells-per-column)
  (unless (typep column 'column-leaf)
    (change-class column 'column-leaf
                  :neurons (map-into (make-array cells-per-column) #'make-neuron))))

(defun surprise-factor (leafs-count bursting-leafs-count)
  "Calculates surprise-factor based on the ratio of leafs not predicted by neurons to all active leafs."
  (declare (type fixnum leafs-count bursting-leafs-count))
  (assert (not (zerop leafs-count)))
  (coerce (/ bursting-leafs-count leafs-count) 'single-float))

(defun handle-leafs (htm leafs)
  "Run one temporal step of the HTM pipeline over LEAFS.

LEAFS is the vector of leaves (one per tree) spatially pooled for a single
data point. Selects the active leafs and ensures their columns are
initialized; propagates distal activity from previously-active neurons;
resolves column activation, bursting any column in which no neuron exceeded
the activation threshold. Computes the surprise factor for this step
(bursting/active ratio), then applies false-positive decay and Hebbian
reinforcement of active synapses, and rotates the active-neuron state for
the next step.

Returns the surprise factor of this step as a single-float."
  (bind ((parameters (htm-parameters htm))
         (state (htm-state htm))
         ((:accessors previously-active-neurons currently-active-neurons) state)
         (activity-hash-table (activity-hash-table state))
         (weights (weights htm))
         (num-trees (length leafs))
         (active-leafs (active-leafs state)))
    (clrhash activity-hash-table)
    (iterate
      (with cells-per-column = (cells-per-column parameters))
      (for i from 0 below num-trees)
      (for leaf = (aref leafs i))
      (setf (gethash leaf active-leafs) t)
      (ensure-column-initialization leaf cells-per-column))
    ;; For each previously-active source neuron, propagate distal activity
    ;; to target neurons that have weights from it (through all connected
    ;; leaves; spatial filtering happens in resolve-column-activation)
    (compute-distal-activation weights previously-active-neurons activity-hash-table)
    (bind ((bursting-count (resolve-column-activation active-leafs
                                                      activity-hash-table
                                                      currently-active-neurons
                                                      parameters))
           ;; Surprise-factor based on bursting ratio, before we reset state
           (surprise-factor (surprise-factor (hash-table-count active-leafs) bursting-count)))
      (decay-false-positive-weights weights previously-active-neurons currently-active-neurons parameters)
      (reinforce-active-synapses previously-active-neurons currently-active-neurons active-leafs weights parameters)
      ;; Prepare state for the next cycle
      (rotatef currently-active-neurons previously-active-neurons)
      (clrhash currently-active-neurons)
      (clrhash active-leafs)
      surprise-factor)))

(defun handle-input (htm data)
  "Process DATA through the HTM temporal pipeline, one time step per row.

DATA is a data matrix (or simple array of rows) in sequence order; each
row is one observation. The ensemble spatially pools every row into one
leaf per tree; rows are then fed to HANDLE-LEAFS in order, which updates
distal activity, resolves column activation, applies the Hebbian weight
update and rotates internal state for the next step.

Returns a (rows x 1) single-float data matrix whose i-th element is the
surprise factor of time step i."
  (bind ((ensemble (htm-ensemble htm))
         (data (sl.data:wrap data))
         (leafs (the sl.data:universal-data-matrix
                     (sl.ensemble:leafs ensemble data)))
         (dims (sl.data:data-matrix-dimensions leafs))
         (num-results (first dims))
         (result (sl.data:make-data-matrix num-results 1)))
    (iterate
      (for i from 0 below num-results)
      (for point-leafs = (sl.data:mref leafs i 0))
      (setf (sl.data:mref result i 0) (handle-leafs htm point-leafs)))
    result))

(defun reset-state (htm &aux (state (htm-state htm)))
  (clrhash (currently-active-neurons state))
  (clrhash (previously-active-neurons state)))
