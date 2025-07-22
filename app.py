# app.py - Flask Backend for Sorting Algorithm Visualizer

from flask import Flask, request, jsonify
import os
from flask import Flask, render_template, request, jsonify, g
from flask_cors import CORS
import uuid
import random
import time

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for frontend communication

# Dictionary to store active sorting sessions.
# Each session will hold:
# - 'generator': The generator object for the chosen sorting algorithm.
# - 'history': A list of states (snapshots) of the array at each step.
# - 'current_step_index': The index of the current state in the history.
sessions = {}

# --- Helper Function for State Representation ---
def _get_current_state(arr, comparisons=None, swaps=None, sorted_indices=None, active_pivot=None, message="", is_finished=False, current_digit_pass=None, current_bucket_elements=None):
    """
    Standardizes the format of the state yielded by sorting algorithms.
    Args:
        arr (list): The current state of the array.
        comparisons (list): Indices of elements currently being compared.
        swaps (list): Indices of elements currently being swapped.
        sorted_indices (list): Indices of elements that are in their final sorted position.
        active_pivot (int): Index of the active pivot element (for Quick Sort).
        message (str): A descriptive message for the current step.
        is_finished (bool): True if the sorting is complete.
        current_digit_pass (int): The current digit place being processed (e.g., 1 for units, 10 for tens).
        current_bucket_elements (list of tuples): (value, original_index) for elements in current bucket processing.
    Returns:
        dict: A dictionary representing the current state.
    """
    return {
        "array": list(arr),  # Make a copy to avoid modifying the original list reference
        "comparisons": comparisons if comparisons is not None else [],
        "swaps": swaps if swaps is not None else [],
        "sorted_indices": sorted_indices if sorted_indices is not None else [],
        "active_pivot": active_pivot,
        "message": message,
        "is_finished": is_finished,
        "current_digit_pass": current_digit_pass,
        "current_bucket_elements": current_bucket_elements if current_bucket_elements is not None else []
    }

# --- Sorting Algorithm Generators ---

def bubble_sort_generator(arr):
    """
    Generator for Bubble Sort visualization.
    Yields the array state and relevant indices at each step.
    """
    n = len(arr)
    sorted_indices = []
    for i in range(n - 1):
        swapped = False
        for j in range(0, n - i - 1):
            # Yield state before comparison
            yield _get_current_state(arr, comparisons=[j, j + 1], message=f"Comparing {arr[j]} and {arr[j+1]}")
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
                # Yield state after swap
                yield _get_current_state(arr, swaps=[j, j + 1], message=f"Swapped {arr[j+1]} and {arr[j]}")
        sorted_indices.append(n - 1 - i) # Mark the largest element as sorted
        if not swapped:
            # If no two elements were swapped by inner loop, array is sorted
            break
    # Mark all elements as sorted at the end
    yield _get_current_state(arr, sorted_indices=list(range(n)), is_finished=True, message="Bubble Sort finished!")


def insertion_sort_generator(arr):
    """
    Generator for Insertion Sort visualization.
    Yields the array state and relevant indices at each step.
    """
    n = len(arr)
    sorted_indices = [] # Elements to the left of 'i' are considered sorted
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        # Yield state before starting inner loop, highlighting key
        yield _get_current_state(arr, comparisons=[i], message=f"Picking {key} (element at index {i}) as key")

        while j >= 0 and key < arr[j]:
            # Yield state during comparison in inner loop
            yield _get_current_state(arr, comparisons=[j, j+1], message=f"Comparing {key} with {arr[j]}")
            arr[j + 1] = arr[j]
            # Yield state after shifting
            yield _get_current_state(arr, swaps=[j, j+1], message=f"Shifting {arr[j+1]} to the right")
            j -= 1
        arr[j + 1] = key
        # Yield state after placing key
        yield _get_current_state(arr, swaps=[j+1, i], message=f"Placing {key} at its correct position")
        sorted_indices.append(i) # Mark elements up to 'i' as part of the sorted sub-array
    yield _get_current_state(arr, sorted_indices=list(range(n)), is_finished=True, message="Insertion Sort finished!")


def merge_sort_generator(arr):
    """
    Generator for Merge Sort visualization.
    Yields the array state and relevant indices at each step.
    """
    # This generator will manage the recursive calls and yield states
    # A list to store all states generated during the sort
    all_states = []

    def merge_sort_recursive(arr_slice, start, end):
        if start >= end:
            return

        mid = (start + end) // 2
        merge_sort_recursive(arr_slice, start, mid)
        merge_sort_recursive(arr_slice, mid + 1, end)
        merge(arr_slice, start, mid, end)

    def merge(arr_slice, start, mid, end):
        left_half = arr_slice[start:mid + 1]
        right_half = arr_slice[mid + 1:end + 1]

        i = j = 0
        k = start

        # Yield state before merging, showing the two sub-arrays
        all_states.append(_get_current_state(
            arr_slice,
            comparisons=list(range(start, end + 1)),
            message=f"Merging sub-arrays from index {start} to {mid} and {mid+1} to {end}"
        ))

        while i < len(left_half) and j < len(right_half):
            # Yield state during comparison in merge
            all_states.append(_get_current_state(
                arr_slice,
                comparisons=[start + i, mid + 1 + j],
                message=f"Comparing {left_half[i]} and {right_half[j]}"
            ))
            if left_half[i] <= right_half[j]:
                arr_slice[k] = left_half[i]
                i += 1
            else:
                arr_slice[k] = right_half[j]
                j += 1
            # Yield state after placing an element
            all_states.append(_get_current_state(
                arr_slice,
                swaps=[k], # Indicate element placed
                message=f"Placing {arr_slice[k]} at index {k}"
            ))
            k += 1

        while i < len(left_half):
            arr_slice[k] = left_half[i]
            all_states.append(_get_current_state(
                arr_slice,
                swaps=[k],
                message=f"Placing remaining {arr_slice[k]} from left half"
            ))
            i += 1
            k += 1

        while j < len(right_half):
            arr_slice[k] = right_half[j]
            all_states.append(_get_current_state(
                arr_slice,
                swaps=[k],
                message=f"Placing remaining {arr_slice[k]} from right half"
            ))
            j += 1
            k += 1
        # Yield state after a merge is complete
        all_states.append(_get_current_state(
            arr_slice,
            sorted_indices=list(range(start, end + 1)),
            message=f"Sub-array from {start} to {end} merged"
        ))

    # Initial call to the recursive function
    temp_arr = list(arr) # Work on a copy to track changes
    merge_sort_recursive(temp_arr, 0, len(temp_arr) - 1)

    # Now, yield all collected states
    for state in all_states:
        yield state
    yield _get_current_state(arr, sorted_indices=list(range(len(arr))), is_finished=True, message="Merge Sort finished!")


def quick_sort_generator(arr):
    """
    Generator for Quick Sort visualization.
    Yields the array state and relevant indices at each step.
    """
    # This generator will manage the recursive calls and yield states
    # A list to store all states generated during the sort
    all_states = []

    def partition(arr_slice, low, high):
        pivot = arr_slice[high]
        all_states.append(_get_current_state(
            arr_slice,
            active_pivot=high,
            message=f"Choosing pivot: {pivot} at index {high}"
        ))
        i = (low - 1)
        for j in range(low, high):
            all_states.append(_get_current_state(
                arr_slice,
                comparisons=[j, high],
                active_pivot=high,
                message=f"Comparing {arr_slice[j]} with pivot {pivot}"
            ))
            if arr_slice[j] <= pivot:
                i += 1
                arr_slice[i], arr_slice[j] = arr_slice[j], arr_slice[i]
                all_states.append(_get_current_state(
                    arr_slice,
                    swaps=[i, j],
                    active_pivot=high,
                    message=f"Swapping {arr_slice[j]} and {arr_slice[i]}"
                ))

        arr_slice[i + 1], arr_slice[high] = arr_slice[high], arr_slice[i + 1]
        all_states.append(_get_current_state(
            arr_slice,
            swaps=[i + 1, high],
            active_pivot=i + 1,
            message=f"Placing pivot {pivot} at its final position {i+1}"
        ))
        return (i + 1)

    def quick_sort_recursive(arr_slice, low, high):
        if len(arr_slice) == 1:
            return arr_slice
        if low < high:
            pi = partition(arr_slice, low, high)
            quick_sort_recursive(arr_slice, low, pi - 1)
            quick_sort_recursive(arr_slice, pi + 1, high)

    # Initial call to the recursive function
    temp_arr = list(arr) # Work on a copy to track changes
    quick_sort_recursive(temp_arr, 0, len(temp_arr) - 1)

    # Now, yield all collected states
    for state in all_states:
        yield state
    yield _get_current_state(arr, sorted_indices=list(range(len(arr))), is_finished=True, message="Quick Sort finished!")


def selection_sort_generator(arr):
    """
    Generator for Selection Sort visualization.
    Yields the array state and relevant indices at each step.
    """
    n = len(arr)
    sorted_indices = []
    for i in range(n):
        min_idx = i
        yield _get_current_state(arr, comparisons=[i], message=f"Assuming {arr[i]} at index {i} is minimum in unsorted part.")
        for j in range(i + 1, n):
            yield _get_current_state(arr, comparisons=[min_idx, j], message=f"Comparing {arr[min_idx]} with {arr[j]}.")
            if arr[j] < arr[min_idx]:
                min_idx = j
                yield _get_current_state(arr, comparisons=[min_idx], message=f"New minimum found: {arr[min_idx]} at index {min_idx}.")
        # Swap the found minimum element with the first element of the unsorted part
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            yield _get_current_state(arr, swaps=[i, min_idx], message=f"Swapping {arr[min_idx]} and {arr[i]} to place minimum.")
        sorted_indices.append(i)
        yield _get_current_state(arr, sorted_indices=list(range(i + 1)), message=f"Element {arr[i]} at index {i} is now sorted.")
    yield _get_current_state(arr, sorted_indices=list(range(n)), is_finished=True, message="Selection Sort finished!")


def radix_sort_generator(arr):
    """
    Generator for Radix Sort visualization.
    Yields the array state and relevant indices at each step.
    Assumes non-negative integers.
    """
    if not arr:
        yield _get_current_state(arr, is_finished=True, message="Array is empty.")
        return

    max_val = max(arr)
    place = 1 # Current digit place (units, tens, hundreds, ...)

    while max_val // place > 0:
        yield _get_current_state(arr, message=f"Starting pass for digits at {place}'s place.", current_digit_pass=place)

        # Perform counting sort for the current digit place
        n = len(arr)
        output = [0] * n
        count = [0] * 10 # 0-9 for digits

        # Store count of occurrences in count[]
        for i in range(n):
            index = (arr[i] // place) % 10
            count[index] += 1
            yield _get_current_state(arr, comparisons=[i], message=f"Counting {arr[i]}'s digit at {place}'s place: {index}", current_digit_pass=place)

        # Change count[i] so that count[i] now contains actual position of this digit in output[]
        for i in range(1, 10):
            count[i] += count[i - 1]

        # Build the output array
        i = n - 1
        while i >= 0:
            index = (arr[i] // place) % 10
            # Highlight element being placed into output
            yield _get_current_state(arr, current_bucket_elements=[(arr[i], i)], message=f"Placing {arr[i]} based on digit {index} into sorted position.", current_digit_pass=place)
            output[count[index] - 1] = arr[i]
            count[index] -= 1
            i -= 1

        # Copy the output array to arr[], so that arr[] now contains sorted numbers according to current digit
        for i in range(n):
            arr[i] = output[i]
            yield _get_current_state(arr, swaps=[i], message=f"Updating array with sorted elements for {place}'s place.", current_digit_pass=place)

        place *= 10
    yield _get_current_state(arr, sorted_indices=list(range(len(arr))), is_finished=True, message="Radix Sort finished!")


# --- Flask Routes ---

@app.route('/start_sort', methods=['POST'])
def start_sort():
    """
    Initializes a new sorting session.
    Expects JSON with 'algorithm' (str) and 'data' (list of int).
    Returns a session ID and the first state of the visualization.
    """
    data = request.json
    algorithm_name = data.get('algorithm')
    input_data_str = data.get('data')

    try:
        # Parse input data string into a list of integers
        input_array = [int(x.strip()) for x in input_data_str.split(',') if x.strip()]
        if not input_array:
            return jsonify({"error": "Input data cannot be empty."}), 400
        # Radix sort only works for non-negative integers
        if algorithm_name == "radix_sort" and any(x < 0 for x in input_array):
            return jsonify({"error": "Radix Sort only supports non-negative integers."}), 400

    except ValueError:
        return jsonify({"error": "Invalid data format. Please enter comma-separated numbers."}), 400

    # Map algorithm names to their generator functions
    algorithm_map = {
        "bubble_sort": bubble_sort_generator,
        "insertion_sort": insertion_sort_generator,
        "merge_sort": merge_sort_generator,
        "quick_sort": quick_sort_generator,
        "selection_sort": selection_sort_generator, # Added Selection Sort
        "radix_sort": radix_sort_generator # Added Radix Sort
    }

    if algorithm_name not in algorithm_map:
        return jsonify({"error": "Invalid algorithm selected."}), 400

    session_id = str(uuid.uuid4())
    generator = algorithm_map[algorithm_name](list(input_array)) # Pass a copy of the array

    # Get the first state and store it in history
    try:
        first_state = next(generator)
    except StopIteration:
        # Handle case where array is already sorted or too small
        first_state = _get_current_state(input_array, sorted_indices=list(range(len(input_array))), is_finished=True, message="Array already sorted or too small.")

    sessions[session_id] = {
        'generator': generator,
        'history': [first_state],
        'current_step_index': 0
    }
    return jsonify({"session_id": session_id, "state": first_state})

@app.route('/next_step/<session_id>', methods=['GET'])
def next_step(session_id):
    """
    Retrieves the next step in the sorting visualization for a given session.
    """
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found."}), 404

    # If we are at the end of the history, try to generate a new step
    if session['current_step_index'] == len(session['history']) - 1:
        try:
            next_state = next(session['generator'])
            session['history'].append(next_state)
            session['current_step_index'] += 1
        except StopIteration:
            # Algorithm has finished, return the last state and mark as finished
            last_state = session['history'][-1]
            if not last_state.get('is_finished'):
                last_state['is_finished'] = True
                last_state['message'] = "Sorting finished!"
            return jsonify({"state": last_state})
    else:
        # Move forward in history
        session['current_step_index'] += 1
        next_state = session['history'][session['current_step_index']]

    return jsonify({"state": next_state})

@app.route('/prev_step/<session_id>', methods=['GET'])
def prev_step(session_id):
    """
    Retrieves the previous step in the sorting visualization for a given session.
    """
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found."}), 404

    if session['current_step_index'] > 0:
        session['current_step_index'] -= 1
        prev_state = session['history'][session['current_step_index']]
        return jsonify({"state": prev_state})
    else:
        # Already at the first step
        return jsonify({"error": "No previous steps available.", "state": session['history'][0]}), 400

@app.route('/reset_session/<session_id>', methods=['POST'])
def reset_session(session_id):
    """
    Resets or clears a specific sorting session.
    """
    if session_id in sessions:
        del sessions[session_id]
        return jsonify({"message": f"Session {session_id} reset successfully."}), 200
    return jsonify({"error": "Session not found."}), 404

# --- Main execution ---
if __name__ == '__main__':
    # Get the port from the environment variable, default to 8080 for local testing
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
