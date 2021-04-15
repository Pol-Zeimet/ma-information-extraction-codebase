import git


class Repository:
    @staticmethod
    def get_details() -> dict:
        repo = git.Repo(search_parent_directories=True)

        return {
            "GIT_DIR": repo.git_dir,
            "GIT_BRANCH": repo.active_branch.name,
            "GIT_COMMIT": repo.head.object.hexsha
        }
