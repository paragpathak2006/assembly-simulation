from flask import request, jsonify, send_file
from flask_cors import CORS
import cgi
import os
import json
import tempfile
import shutil
import logging
import requests
import uuid
import time
from celery import group, chord

from . import app, celery

from .sequence_planner import parallel_sequence_planner, finish_buildit
from .lib.convert import run_conversion
# from .lib.subdivide import subdivide_assembly
# from .lib.subassembly_generator import generate_subassemblies

CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

@app.route('/', methods=['GET'])
def not_allowed():
    return jsonify({'message': 'You are not allowed to access this site'}), 405

@app.route('/health', methods=['GET'])
def health_check():
    return "OK", 200

@app.route('/upload_assembly', methods=['POST'])
def upload_assembly():

    logging.info("Doing POST")
    content_type = request.headers.get('Content-Type')
    ctype, pdict = cgi.parse_header(content_type)

    # refuse to receive non-json content
    if ctype != 'multipart/form-data':
        return jsonify({'message': 'Bad request'}), 400

    try:
        form_data = request.files
        max_file_size_mb = 50
        max_file_size_bytes = max_file_size_mb * 1024 * 1024
        step_file = form_data['stepFile']
        chunk_size = 4096

        bytes_read = 0
        step_file_data = b""
        while True:
            chunk = step_file.stream.read(chunk_size)
            if len(chunk) == 0:
                break

            bytes_read += len(chunk)
            if bytes_read > max_file_size_bytes:
                return jsonify({'message': 'File size exceeds 50MB'}), 400

            step_file_data += chunk

        step_file_data = step_file_data.decode('utf-8')
    except Exception as err:
        logging.error("Could not parse input file:" + str(err))
        return jsonify({'message': 'The file cannot be parsed. Please check the format and try again.'}), 400

    proposed_disassembly_order = None
    if 'proposedOrder' in request.form:
        proposed_assembly_order = json.loads(request.form['proposedOrder'])
        proposed_disassembly_order = []
        for subassembly in proposed_assembly_order:
            proposed_disassembly_order.append(subassembly[::-1])

    presigned_url = request.form['presignedUrl']
    task = process_assembly_async.delay(step_file_data, presigned_url, proposed_disassembly_order)
    return jsonify({"task_id": task.id}), 202

@celery.task
def process_subassembly(run_dir, subassembly_idx, subassembly, proposed_orders):
    try:
        return parallel_sequence_planner(run_dir, subassembly_idx, subassembly, proposed_orders)
    except Exception as err:
        logging.error("buildit Error: " + str(err))
        return({"statusCode": 500, "message": str(err)})

@celery.task
def finalize_processing(sequence_planner_results, runFilepath, part_names, part_indices, presigned_url, outputFilepath, temp_dir):
    try:
        output = finish_buildit(sequence_planner_results, runFilepath, part_names, part_indices)

        with open(os.path.join(runFilepath, 'solution.json'), 'w') as solution_file:
            json.dump(output, solution_file)

        shutil.make_archive(os.path.join(outputFilepath, 'output'), 'zip', runFilepath)

        # Upload the resulting ZIP file to S3
        try :
            upload_to_s3(os.path.join(outputFilepath, 'output.zip'), presigned_url)
        except Exception as err:
            return {"statusCode": 500, "message": "Could not upload result to S3"}

        shutil.rmtree(temp_dir)
        return {"statusCode": 200, "message": "Success"}

    except Exception as err:
        logging.error("buildit Error: " + str(err))
        shutil.rmtree(temp_dir)
        return({"statusCode": 500, "message": str(err)})

@celery.task(bind=True)
def process_assembly_async(self, step_file_data, presigned_url, proposed_disassembly_order):
    temp_dir = os.path.join("/home/ubuntu/buildit", str(uuid.uuid4()))
    try:
        inputFilepath = os.path.join(temp_dir, "input")
        inputStepFile = os.path.join(inputFilepath, "input.step")
        runFilepath = os.path.join(temp_dir, "run")
        outputFilepath = os.path.join(temp_dir, "output")
        os.makedirs(inputFilepath, exist_ok=False)
        os.makedirs(runFilepath, exist_ok=False)
        os.makedirs(outputFilepath, exist_ok=False)
        with open(inputStepFile, 'w') as input_file:
            input_file.write(step_file_data)

        try:
            subassemblies, part_names, part_indices = run_conversion(inputStepFile, runFilepath)
        except Exception as err:
            logging.error("Could not convert files")
            return({"statusCode": 400, "message": "Could not process CAD file. Please check your file."})

        if (len(part_indices) == 1):
            logging.error("File only has one part. Killing buildit.")
            return({"statusCode": 400, "message": "Invalid CAD file. The file should have multiple parts."})

        subassembly_order = [[subassemblies[subassembly_idx].index(indices) for indices in subassembly_order] for subassembly_idx, subassembly_order in enumerate(proposed_disassembly_order)] if proposed_disassembly_order else None

        # Create a group of tasks, one for each subassembly
        task_group = group(process_subassembly.s(runFilepath, subassembly_idx, subassembly, subassembly_order) 
                for subassembly_idx, subassembly in enumerate(subassemblies))

        # Create a chord with the group of tasks and a callback
        callback = finalize_processing.s(runFilepath, part_names, part_indices, presigned_url, outputFilepath, temp_dir)
        chord_result = chord(task_group)(callback)

        # Block until the callback task has finished
        while not chord_result.ready():
            time.sleep(1)

        return chord_result.result

    except Exception as err:
        logging.error("buildit Error: " + str(err))
        shutil.rmtree(temp_dir)
        return({"statusCode": 500, "message": str(err)})

def upload_to_s3(file_path, presigned_url):
    with open(file_path, 'rb') as f:
        headers = {'Content-Type': 'application/zip'}
        response = requests.put(presigned_url, data=f, headers=headers)
        if response.status_code != 200:
            logging.error(response)
            raise ValueError('Failed to upload file to S3')
        return response.status_code

@app.route('/task_status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task = process_assembly_async.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Task is processing'
        }
    else:
        response = {
            'state': task.state,
            'status': task.info
        }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
