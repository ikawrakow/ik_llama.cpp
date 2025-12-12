#pragma once
#include "server-task.h"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <unordered_set>

struct server_task_multi {
    int id = -1;

    std::set<int> subtasks_remaining;
    std::vector<server_task_result> results;
};


struct server_queue {
    int id = 0;
    bool running;

    // queues
    std::vector<server_task> queue_tasks;
    std::vector<server_task> queue_tasks_deferred;

    std::vector<server_task_multi> queue_multitasks;

    std::mutex mutex_tasks;
    std::condition_variable condition_tasks;

    // callback functions
    std::function<void(server_task &&)> callback_new_task;
    std::function<void(server_task_multi &)> callback_finish_multitask;
    std::function<void(void)>                callback_update_slots;


    // Add a new task to the end of the queue
    int post(server_task task);

    // Add a new task, but defer until one slot is available
    void defer(server_task&& task);

    // Get the next id for creating anew task
    int get_new_id();

    // Register function to process a new task
    void on_new_task(std::function<void(server_task&&)> callback);

    // Register function to process a multitask when it is finished
    void on_finish_multitask(std::function<void(server_task_multi&)> callback) {
        callback_finish_multitask = std::move(callback);
    }

    // Register the function to be called when all slots data is ready to be processed
    void on_update_slots(std::function<void(void)> callback) {
        callback_update_slots = std::move(callback);
    }

    // Call when the state of one slot is changed
    void notify_slot_changed();

    // end the start_loop routine
    void terminate() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        running = false;
        condition_tasks.notify_all();
    }

    /**
     * Main loop consists of these steps:
     * - Wait until a new task arrives
     * - Process the task (i.e. maybe copy data into slot)
     * - Check if multitask is finished
     * - Update all slots
     */
    void start_loop();

    //
    // functions to manage multitasks
    //

    // add a multitask by specifying the id of all subtask (subtask is a server_task)
    void add_multitask(int id_multi, std::vector<int>& sub_ids);

    // updatethe remaining subtasks, while appending results to multitask
    void update_multitask(int id_multi, int id_sub, server_task_result& result);
};

struct server_response {
    typedef std::function<void(int, int, server_task_result&)> callback_multitask_t;
    callback_multitask_t callback_update_multitask;

    // for keeping track of all tasks waiting for the result
    std::set<int> waiting_task_ids;

    // the main result queue
    std::vector<server_task_result> queue_results;

    std::mutex mutex_results;
    std::condition_variable condition_results;

    // add the id_task to the list of tasks waiting for response
    void add_waiting_task_id(int id_task);

    // when the request is finished, we can remove task associated with it
    void remove_waiting_task_id(int id_task);

    // This function blocks the thread until there is a response for this id_task
    server_task_result recv(int id_task);

    // Register the function to update multitask
    void on_multitask_update(callback_multitask_t callback) {
        callback_update_multitask = std::move(callback);
    }

    // Send a new result to a waiting id_task
    void send(server_task_result result);
};
