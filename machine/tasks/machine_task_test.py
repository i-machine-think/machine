from machine.tasks import get_task

task_a = get_task("long_lookup_reverse")
task_b = get_task("lookup")
task_c = get_task("symbol_rewriting")

print(task_a)
print(task_b)
print(task_c)
