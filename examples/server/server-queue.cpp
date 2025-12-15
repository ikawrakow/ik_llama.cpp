#include "server-task.h"
#include "server-queue.h"
#include "server-common.h"

#include "log.h"
#include <chrono>

#define QUE_INF(fmt, ...) LOG_INF("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_WRN(fmt, ...) LOG_WRN("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...) LOG_DBG("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

#define RES_INF(fmt, ...) LOG_INF("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_WRN(fmt, ...) LOG_WRN("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_ERR(fmt, ...) LOG_ERR("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_DBG(fmt, ...) LOG_DBG("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)


int server_queue::post(server_task task) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (task.id == -1) {
        task.id = id++;
        //LOG_VERBOSE("new task id", { {"new_id", task.id} });
        QUE_DBG("new task, id = %d\n", task.id);
    }
    queue_tasks.push_back(std::move(task));
    condition_tasks.notify_one();
    return task.id;
}

void server_queue::defer(server_task&& task) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    queue_tasks_deferred.push_back(std::move(task));
}

int server_queue::get_new_id() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    int new_id = id++;
    //LOG_VERBOSE("new task id", { {"new_id", new_id} });
    QUE_DBG("new task, id = %d\n", id);
    return new_id;
}

void server_queue::notify_slot_changed() {
    // move deferred tasks back to main loop
    std::unique_lock<std::mutex> lock(mutex_tasks);
    for (auto& task : queue_tasks_deferred) {
        queue_tasks.push_back(std::move(task));
    }
    queue_tasks_deferred.clear();
}

void server_queue::on_new_task(std::function<void(server_task&&)> callback) {
    callback_new_task = std::move(callback);
}


void server_queue::start_loop() {
    running = true;

    while (true) {
        LOG_VERBOSE("new task may arrive", {});

        while (true) {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (queue_tasks.empty()) {
                lock.unlock();
                break;
            }
            server_task task = std::move(queue_tasks.front());
            queue_tasks.erase(queue_tasks.begin());
            lock.unlock();
            //LOG_VERBOSE("callback_new_task", { {"id_task", task.id} });
            callback_new_task(std::move(task));
        }

        LOG_VERBOSE("update_multitasks", {});

        // check if we have any finished multitasks
        auto queue_iterator = queue_multitasks.begin();
        while (queue_iterator != queue_multitasks.end()) {
            if (queue_iterator->subtasks_remaining.empty()) {
                // all subtasks done == multitask is done
                server_task_multi current_multitask = *queue_iterator;
                callback_finish_multitask(current_multitask);
                // remove this multitask
                queue_iterator = queue_multitasks.erase(queue_iterator);
            }
            else {
                ++queue_iterator;
            }
        }

        // all tasks in the current loop is processed, slots data is now ready
        LOG_VERBOSE("callback_update_slots", {});

        callback_update_slots();

        LOG_VERBOSE("wait for new task", {});
        {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (queue_tasks.empty()) {
                if (!running) {
                    LOG_VERBOSE("ending start_loop", {});
                    return;
                }
                condition_tasks.wait(lock, [&] {
                    return (!queue_tasks.empty() || !running);
                    });
            }
        }
    }
}


void server_queue::add_multitask(int id_multi, std::vector<int>& sub_ids) {
    std::lock_guard<std::mutex> lock(mutex_tasks);
    server_task_multi multi;
    multi.id = id_multi;
    std::copy(sub_ids.begin(), sub_ids.end(), std::inserter(multi.subtasks_remaining, multi.subtasks_remaining.end()));
    queue_multitasks.push_back(multi);
}


void server_queue::update_multitask(int id_multi, int id_sub, server_task_result& result) {
    std::lock_guard<std::mutex> lock(mutex_tasks);
    for (auto& multitask : queue_multitasks) {
        if (multitask.id == id_multi) {
            multitask.subtasks_remaining.erase(id_sub);
            multitask.results.push_back(result);
        }
    }
}


void server_response::add_waiting_task_id(int id_task) {
    //LOG_VERBOSE("waiting for task id", { {"id_task", id_task} });
    QUE_DBG("waiting for task id, id = %d\n", id_task);

    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.insert(id_task);
}

void server_response::remove_waiting_task_id(int id_task) {
    //LOG_VERBOSE("remove waiting for task id", { {"id_task", id_task} });
    QUE_DBG("remove waiting for task id, id = %d\n", id_task);
    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.erase(id_task);
}


server_task_result server_response::recv(int id_task) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);
        condition_results.wait(lock, [&] {
            return !queue_results.empty();
            });

        for (int i = 0; i < (int)queue_results.size(); i++) {
            if (queue_results[i].id == id_task) {
                assert(queue_results[i].id_multi == -1);
                server_task_result res = queue_results[i];
                queue_results.erase(queue_results.begin() + i);
                return res;
            }
        }
    }

    // should never reach here
}

void server_response::send(server_task_result result) {
    //LOG_VERBOSE("send new result", { {"id_task", result.id} });
    QUE_DBG("send new result, id = %d\n", result.id);
    std::unique_lock<std::mutex> lock(mutex_results);
    for (const auto& id_task : waiting_task_ids) {
        // LOG_TEE("waiting task id %i \n", id_task);
        // for now, tasks that have associated parent multitasks just get erased once multitask picks up the result
        if (result.id_multi == id_task) {
            //LOG_VERBOSE("callback_update_multitask", { {"id_task", id_task} });
            QUE_DBG("callback_update_multitask, id = %d\n", id_task);
            callback_update_multitask(id_task, result.id, result);
            continue;
        }

        if (result.id == id_task) {
            //LOG_VERBOSE("queue_results.push_back", { {"id_task", id_task} });
            QUE_DBG("queue_results.push_back, id = %d\n", id_task);
            queue_results.push_back(result);
            condition_results.notify_all();
            return;
        }
    }
}
